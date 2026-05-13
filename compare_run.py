#!/usr/bin/env python3
"""
Q-Learning vs SARSA vs DQN — warehouse delivery robot, side-by-side.

Run:  python3 compare_run.py

Each panel is a top-down warehouse: wood-plank floor, wooden crates as
obstacles, a charger, a cardboard parcel at the goal, and a cute robot
that picks up the box and drops it off (✓ on the destination tile).
Three algorithms run in parallel on different layouts.
"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict, deque

from utils.config import EnvConfig, DQNConfig
from environment.city_env import CityDeliveryEnv
from agents.dqn_agent import DQNAgent as _DQNAgent
from agents.q_learning_agent import QLearningAgent as QLAgent
from agents.sarsa_agent import SARSAAgent

# ══════════════════════════════════════════════════════════════════════
#  CONFIG  —  warehouse delivery
# ══════════════════════════════════════════════════════════════════════

ENV_CFG = EnvConfig(
    grid_size               = (10, 10),
    num_deliveries          = 1,
    max_steps               = 80,
    fov_radius              = 2,
    max_battery             = 9999,
    num_traffic_zones       = 4,         # 4 patches of warehouse congestion
    traffic_transition_prob = 0.05,      # traffic shifts dynamically each step
    rush_hour_steps         = [],
    wall_density            = 0.10,
    reward_delivery         = 100.0,
    reward_in_time          = 50.0,
    penalty_step            = -0.5,
    penalty_traffic         = -5.0,      # per-step penalty when on a traffic cell
    penalty_timeout         = -50.0,
    penalty_dead_battery    = 0.0,
    reward_battery_pickup   = 0.0,
)

DQN_CFG = DQNConfig(
    hidden_dims         = [128, 128],
    lr                  = 1e-3,
    gamma               = 0.95,
    epsilon_start       = 1.0,
    epsilon_end         = 0.06,
    epsilon_decay_steps = 4000,
    batch_size          = 32,
    buffer_size         = 8_000,
    use_per             = False,
    use_soft_update     = False,
    target_update_freq  = 150,
    grad_clip           = 5.0,
)

DQN_DEMO_CHECKPOINT  = "checkpoints/compare_dqn_v6.pt"
TABULAR_PRETRAIN_EPS = 280
DQN_PRETRAIN_EPS     = 600
DEMO_EPSILON         = 0.05
SMOOTH_FRAMES        = 2
RENDER_PAUSE         = 0.005

# Disjoint pools so DQN's FOV-based generalization shows up vs tabular memorization.
TRAIN_SEEDS = list(range(0, 30))
DEMO_SEEDS  = list(range(50, 100))

HDR_COLS = ["#1a3a6b", "#6b1a1a", "#1a5c2a"]
BOT_COLS = ["#4a90d9", "#d94a5a", "#4ad97a"]
LABELS   = ["Q-Learning", "SARSA", "DQN"]

ACT_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_env(seed=0):
    return CityDeliveryEnv(ENV_CFG, difficulty=2, seed=seed)


def bfs_dist(pos, goal, walls, grid_size):
    if pos == goal: return 0
    H, W = grid_size
    visited = {pos}
    q = [(pos, 0)]
    while q:
        (r, c), d = q.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (r+dr, c+dc)
            if nb == goal: return d+1
            if 0<=nb[0]<H and 0<=nb[1]<W and nb not in visited and nb not in walls:
                visited.add(nb); q.append((nb, d+1))
    return 999


def bfs_path(start, goal, walls, grid_size):
    """Return list of actions (0=up,1=down,2=left,3=right) from start to goal,
    or [] if unreachable. Used to break the agent out of stuck loops."""
    if start == goal: return []
    H, W = grid_size
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    parent = {start: None}
    queue = [start]
    found = False
    while queue and not found:
        pos = queue.pop(0)
        for a, (dr, dc) in enumerate(deltas):
            nb = (pos[0] + dr, pos[1] + dc)
            if nb == goal:
                parent[nb] = (pos, a)
                found = True
                break
            if (0 <= nb[0] < H and 0 <= nb[1] < W
                    and nb not in walls and nb not in parent):
                parent[nb] = (pos, a)
                queue.append(nb)
    if not found:
        return []
    out = []
    cur = goal
    while parent[cur] is not None:
        par, act = parent[cur]
        out.append(act)
        cur = par
    out.reverse()
    return out


def disc_state(env):
    ar, ac = env.agent_pos
    gr, gc = (env.delivery_queue[0].destination
              if env.delivery_queue else (ar, ac))
    return (ar, ac, gr, gc)


def shaped(prev_pos, curr_pos, goal, walls, grid_size, dcache,
           traffic_grid=None):
    for p in (prev_pos, curr_pos):
        if (p, goal) not in dcache:
            dcache[(p, goal)] = bfs_dist(p, goal, walls, grid_size)
    s = 8.0 * (dcache[(prev_pos, goal)] - dcache[(curr_pos, goal)])
    if curr_pos == prev_pos: s -= 3.0
    # Traffic-avoidance signal: -4 / -8 / -12 for landing on lvl 1/2/3
    if traffic_grid is not None:
        lvl = int(traffic_grid[curr_pos[0], curr_pos[1]])
        if lvl > 0:
            s -= lvl * 4.0
    return s


# ══════════════════════════════════════════════════════════════════════
#  DQN WRAPPER
#  (QLearningAgent and SARSAAgent live in agents/q_learning_agent.py
#   and agents/sarsa_agent.py respectively — see those files for the
#   tabular update rules.)
# ══════════════════════════════════════════════════════════════════════

class DQNWrapper:
    def __init__(self):
        self._ag     = _DQNAgent(ENV_CFG, DQN_CFG, agent_type="dueling_ddqn")
        self.episode = 0
        self.best    = 0
        self.loaded  = False

    @property
    def epsilon(self): return self._ag.epsilon

    def act(self, obs):
        return self._ag.select_action(obs, greedy=False)

    def store_and_learn(self, obs, action, r, nobs, done):
        self._ag.store(obs, action, r, nobs, done)
        self._ag.learn()

    def on_done(self, raw_score):
        self.best     = max(self.best, raw_score)
        self.episode += 1


# ══════════════════════════════════════════════════════════════════════
#  PRETRAIN
# ══════════════════════════════════════════════════════════════════════

def pretrain_tabular(agent, label, episodes, seeds):
    print(f"  training {label} ", end="", flush=True)
    bar_step = max(1, episodes // 30)
    for ep in range(episodes):
        env = make_env(seeds[ep % len(seeds)])
        obs, _ = env.reset()
        dcache = {}
        ep_raw = 0.0
        while True:
            s = disc_state(env)
            a = agent.act(s)
            prev_pos = env.agent_pos
            goal = (env.delivery_queue[0].destination
                    if env.delivery_queue else prev_pos)
            obs, raw_r, term, trunc, info = env.step(a)
            curr_pos = env.agent_pos
            done = term or trunc
            r = raw_r + shaped(prev_pos, curr_pos, goal,
                               env._walls, env.cfg.grid_size, dcache,
                               traffic_grid=env.traffic.get_grid())
            ns = disc_state(env)
            agent.learn(s, a, r, ns, done)
            ep_raw += raw_r
            if done: break
        agent.on_done(ep_raw)
        if ep % bar_step == 0: print(".", end="", flush=True)
    agent.epsilon = DEMO_EPSILON
    print(" done.")


def pretrain_dqn(wrapper, label, episodes, seeds):
    print(f"  training {label} ", end="", flush=True)
    bar_step    = max(1, episodes // 30)
    LEARN_EVERY = 4
    for ep in range(episodes):
        env = make_env(seeds[ep % len(seeds)])
        obs, _ = env.reset()
        dcache = {}
        steps  = 0
        while True:
            a = wrapper._ag.select_action(obs, greedy=False)
            # Prevent no-op collapse: force movement actions during DQN training.
            if a == 4:
                a = int(np.random.randint(4))
            prev_pos = env.agent_pos
            goal = (env.delivery_queue[0].destination
                    if env.delivery_queue else prev_pos)
            next_obs, raw_r, term, trunc, info = env.step(a)
            curr_pos = env.agent_pos
            done = term or trunc
            r = raw_r + shaped(prev_pos, curr_pos, goal,
                               env._walls, env.cfg.grid_size, dcache,
                               traffic_grid=env.traffic.get_grid())
            wrapper._ag.store(obs, a, r, next_obs, float(done))
            steps += 1
            if steps % LEARN_EVERY == 0:
                wrapper._ag.learn()
            obs = next_obs
            if done: break
        if ep % bar_step == 0: print(".", end="", flush=True)
    wrapper._ag._epsilon = DEMO_EPSILON
    print(" done.")


DQN_MIN_EPS_FOR_CACHE = 400   # only persist a checkpoint if pretrain was substantial

def prepare_dqn(seeds):
    dqn = DQNWrapper()
    if os.path.exists(DQN_DEMO_CHECKPOINT):
        try:
            dqn._ag.load(DQN_DEMO_CHECKPOINT)
            dqn._ag._epsilon = DEMO_EPSILON
            dqn.loaded = True
            print(f"  Loaded DQN cache from {DQN_DEMO_CHECKPOINT}")
            return dqn
        except Exception as e:
            print(f"  DQN cache unreadable ({e}) — retraining.")
    pretrain_dqn(dqn, "DQN        ", DQN_PRETRAIN_EPS, seeds)
    if DQN_PRETRAIN_EPS >= DQN_MIN_EPS_FOR_CACHE:
        os.makedirs(os.path.dirname(DQN_DEMO_CHECKPOINT) or ".", exist_ok=True)
        dqn._ag.save(DQN_DEMO_CHECKPOINT)
        print(f"  Cached DQN to {DQN_DEMO_CHECKPOINT}")
    else:
        print(f"  Pretrain was only {DQN_PRETRAIN_EPS} eps "
              f"(< {DQN_MIN_EPS_FOR_CACHE}) — not caching.")
    return dqn


# ══════════════════════════════════════════════════════════════════════
#  RENDERER  —  warehouse sprites
# ══════════════════════════════════════════════════════════════════════

# Wood-plank base palette (per-tile jitter applied)
WOOD_BASE_FC = (0xc8, 0x90, 0x58)   # warm tan
WOOD_DARK_EC = "#5e3a14"
WOOD_SEAM_C  = "#704018"

def _tile_color(r, c):
    h = (r * 13 + c * 7) % 7
    j = (h - 3) * 6   # ±18
    return "#%02x%02x%02x" % tuple(
        max(0, min(255, b + j)) for b in WOOD_BASE_FC)


def _make_floor_img(H, W, tpx=24):
    """Render the wood-plank floor once as an RGB numpy array.
    Returns a (H*tpx, W*tpx, 3) array suitable for ax.imshow."""
    base = np.array(WOOD_BASE_FC, dtype=float) / 255.0
    img = np.zeros((H * tpx, W * tpx, 3), dtype=float)
    for r in range(H):
        for c in range(W):
            j = (((r * 13 + c * 7) % 7) - 3) * 0.025
            color = np.clip(base + j, 0.0, 1.0)
            img[r*tpx:(r+1)*tpx, c*tpx:(c+1)*tpx] = color
    # Tile boundaries (darker)
    for r in range(H + 1):
        y = min(r * tpx, H * tpx - 1)
        img[y, :] *= 0.55
    for c in range(W + 1):
        x = min(c * tpx, W * tpx - 1)
        img[:, x] *= 0.55
    # Plank seams within each tile (3 horizontal lines per tile)
    for r in range(H):
        for k_frac in (0.25, 0.55, 0.82):
            y = r * tpx + int(tpx * k_frac)
            img[y, :] *= 0.78
    return np.clip(img, 0.0, 1.0)


class CompareRenderer:
    TRAIL = 12

    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(15.5, 8.6), facecolor="#0f1015")
        try:
            self.fig.canvas.manager.set_window_title(
                "Warehouse Delivery Robot — Q-Learning vs SARSA vs DQN")
        except Exception:
            pass

        gs = GridSpec(3, 3, figure=self.fig,
                      height_ratios=[1.0, 5.0, 0.85],
                      hspace=0.06, wspace=0.04,
                      left=0.01, right=0.99,
                      top=0.985, bottom=0.025)

        self.lax = self.fig.add_subplot(gs[0, :])
        self.lax.set_facecolor("#13141c")
        self.lax.set_xticks([]); self.lax.set_yticks([])
        for sp in self.lax.spines.values():
            sp.set_edgecolor("#243046"); sp.set_linewidth(1.2)

        self.gax, self.hax = [], []
        for col in range(3):
            ga = self.fig.add_subplot(gs[1, col])
            ha = self.fig.add_subplot(gs[2, col])
            ga.set_facecolor("#0f1015"); ha.set_facecolor("#13141c")
            for ax in (ga, ha): ax.set_xticks([]); ax.set_yticks([])
            self.gax.append(ga); self.hax.append(ha)

        self.trails    = [deque(maxlen=self.TRAIL) for _ in range(3)]
        self.flashes   = [0, 0, 0]
        self.cur_score = [0.0, 0.0, 0.0]
        self.facing    = [3, 3, 3]
        self.suc_hist  = [deque(maxlen=20) for _ in range(3)]
        self.scr_hist  = [deque(maxlen=20) for _ in range(3)]
        self.stp_hist  = [deque(maxlen=20) for _ in range(3)]
        self.delivered = [0, 0, 0]
        self.popups    = [None, None, None]
        self.checks    = [None, None, None]   # (r, c, frames_remaining)
        H, W = ENV_CFG.grid_size
        self._floor_img = _make_floor_img(H, W)
        self._last_state = [None, None, None]   # static-layout snapshot per panel
        self._dyn        = [[], [], []]         # dynamic artists per panel

    # ─── public hooks ────────────────────────────────────────
    def notify_delivery(self, i, anchor_xy, goal_rc):
        self.flashes[i] = 10
        self.popups[i]  = ["+100", 24, anchor_xy]
        self.checks[i]  = [goal_rc[0], goal_rc[1], 30]
    def reset_trail(self, i):
        self.trails[i].clear()
        self._last_state[i] = None   # force static-layer redraw next frame
    def append_trail(self, i, pos):
        self.trails[i].append((pos[0], pos[1]))
    def set_last_action(self, i, action):
        if action in (0, 1, 2, 3): self.facing[i] = action
    def record_episode(self, i, success, score, steps):
        self.suc_hist[i].append(1 if success else 0)
        self.scr_hist[i].append(score)
        self.stp_hist[i].append(steps)
        if success: self.delivered[i] += 1

    # ─── main render entry ──────────────────────────────────
    def render(self, envs, agents, interp_pos=None, full=True):
        """`full=False` skips the HUD and leaderboard — use for in-between
        interp frames where only the robot's position has actually changed."""
        if full:
            avgs = [(np.mean(h) if h else None) for h in self.scr_hist]
            ranks = [None, None, None]
            if any(a is not None for a in avgs):
                order = sorted(range(3),
                               key=lambda j: (-avgs[j] if avgs[j] is not None else 1e9))
                for rk, j in enumerate(order, start=1):
                    if avgs[j] is not None: ranks[j] = rk
            self._cur_ranks = ranks
        for i in range(3):
            pos = (interp_pos[i] if interp_pos
                   else (float(envs[i].agent_pos[0]),
                         float(envs[i].agent_pos[1])))
            self._panel(i, envs[i], agents[i], pos)
            if full:
                self._hud(i, envs[i], agents[i])
        if full:
            self._leaderboard(agents)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ─── leaderboard (top strip) ────────────────────────────
    def _leaderboard(self, agents):
        ax = self.lax
        ax.clear()
        ax.set_facecolor("#13141c")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for sp in ax.spines.values():
            sp.set_edgecolor("#243046"); sp.set_linewidth(1.2)

        ax.text(0.005, 0.93, "LIVE  LEADERBOARD",
                color="#7a90c0", fontsize=8.5, fontweight="bold",
                fontfamily="monospace", transform=ax.transAxes,
                ha="left", va="top")

        avg_scores = [(np.mean(h) if h else None) for h in self.scr_hist]
        suc_rates  = [(np.mean(h) if h else 0.0)  for h in self.suc_hist]
        avg_steps  = [(np.mean(h) if h else ENV_CFG.max_steps)
                      for h in self.stp_hist]
        ready      = [len(h) >= 2 for h in self.scr_hist]

        scored = [(s if s is not None else -np.inf) for s in avg_scores]
        finite = [s for s in scored if np.isfinite(s)]
        max_sc = max(finite) if finite else 1.0
        leader = -1
        if any(ready):
            ranked = [(s if r else -np.inf) for s, r in zip(scored, ready)]
            leader = int(np.argmax(ranked))

        for i in range(3):
            x0   = 0.020 + i * 0.327
            wbar = 0.305

            ax.add_patch(mpatches.FancyBboxPatch(
                (x0, 0.05), wbar, 0.78,
                boxstyle="round,pad=0.005,rounding_size=0.012",
                fc="#10131b", ec=HDR_COLS[i], lw=1.4,
                transform=ax.transAxes))

            ax.text(x0 + 0.012, 0.74, LABELS[i],
                    color="white", fontsize=11, fontweight="bold",
                    fontfamily="monospace", transform=ax.transAxes,
                    ha="left", va="center")

            if i == leader:
                ax.text(x0 + 0.135, 0.74, "★ LEADER",
                        color="#ffe070", fontsize=9.5, fontweight="bold",
                        fontfamily="monospace", transform=ax.transAxes,
                        ha="left", va="center")

            ax.text(x0 + wbar - 0.012, 0.74,
                    f"EP {agents[i].episode}   ✓ {self.delivered[i]}",
                    color="#aaccee", fontsize=8.5,
                    fontfamily="monospace", transform=ax.transAxes,
                    ha="right", va="center")

            sr = suc_rates[i]
            sc = avg_scores[i] if avg_scores[i] is not None else 0.0
            sc_norm = (sc / max_sc) if max_sc > 0.001 else 0.0
            sc_norm = max(0.0, min(1.0, sc_norm))
            st = avg_steps[i]
            st_norm = max(0.0, 1.0 - st / ENV_CFG.max_steps)

            metrics = [
                ("Success",   sr,      f"{sr*100:>3.0f}%"),
                ("Avg Score", sc_norm, f"{sc:>+5.0f}"),
                ("Speed",     st_norm, f"{st:>4.1f} stp"),
            ]
            for k, (name, val, txt) in enumerate(metrics):
                bx0 = x0 + 0.012
                by0 = 0.54 - k * 0.14
                bw  = wbar - 0.024
                ax.text(bx0, by0 + 0.035, name,
                        color="#5566aa", fontsize=7.5,
                        fontfamily="monospace", transform=ax.transAxes,
                        ha="left", va="center")
                ax.add_patch(mpatches.Rectangle(
                    (bx0 + 0.080, by0 - 0.005), bw - 0.140, 0.05,
                    fc="#101424", ec="#243046", lw=0.6,
                    transform=ax.transAxes))
                fillw = (bw - 0.140) * val
                if fillw > 0.001:
                    ax.add_patch(mpatches.Rectangle(
                        (bx0 + 0.080, by0 - 0.005), fillw, 0.05,
                        fc=BOT_COLS[i], ec="none",
                        transform=ax.transAxes))
                ax.text(x0 + wbar - 0.012, by0 + 0.020, txt,
                        color="white", fontsize=8.5, fontweight="bold",
                        fontfamily="monospace", transform=ax.transAxes,
                        ha="right", va="center")

    # ─── sprite primitives ──────────────────────────────────
    def _draw_floor_fast(self, ax, H, W):
        ax.imshow(self._floor_img,
                  extent=(-0.5, W - 0.5, H - 0.5, -0.5),
                  zorder=1, interpolation="nearest")

    @staticmethod
    def _draw_crate(ax, r, c):
        # Crate body — 1 patch + 2 slat lines
        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.36, r - 0.32), 0.72, 0.66,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            fc="#5a3614", ec="#2c1808", lw=1.4, zorder=4))
        ax.plot([c - 0.32, c + 0.32], [r - 0.10, r - 0.10],
                color="#2c1808", lw=0.9, zorder=5)
        ax.plot([c - 0.32, c + 0.32], [r + 0.12, r + 0.12],
                color="#2c1808", lw=0.9, zorder=5)

    @staticmethod
    def _draw_dropoff(ax, r, c):
        """Green delivery pad with a bullseye target marker."""
        # Outer green pad
        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.44, r - 0.44), 0.88, 0.88,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            fc="#0c5530", ec="#3aee70", lw=2.3, zorder=4))
        # Inner brighter pad
        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.34, r - 0.34), 0.68, 0.68,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            fc="#1ea050", ec="none", zorder=5))
        # Bullseye: outer ring + inner dot
        ax.add_patch(plt.Circle((c, r), 0.22,
                                fc="none", ec="white", lw=1.8, zorder=6))
        ax.add_patch(plt.Circle((c, r), 0.10,
                                fc="white", ec="none", zorder=6))
        ax.add_patch(plt.Circle((c, r), 0.05,
                                fc="#1ea050", ec="none", zorder=7))

    @staticmethod
    def _draw_parcel(ax, r, c, scale=1.0, zorder_base=4):
        w, h = 0.62 * scale, 0.46 * scale
        # Shadow
        ax.add_patch(mpatches.Ellipse(
            (c, r + h * 0.6), w * 1.0, 0.08 * scale,
            fc="#000000", alpha=0.35, zorder=zorder_base - 1))
        # Box
        ax.add_patch(mpatches.FancyBboxPatch(
            (c - w/2, r - h/2), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            fc="#c08550", ec="#704018", lw=1.0, zorder=zorder_base))
        # Tape stripes
        ax.add_patch(mpatches.Rectangle(
            (c - w/2, r - 0.06 * scale), w, 0.12 * scale,
            fc="#9a6028", ec="none", alpha=0.85, zorder=zorder_base + 1))
        ax.add_patch(mpatches.Rectangle(
            (c - 0.05 * scale, r - h/2), 0.10 * scale, h,
            fc="#9a6028", ec="none", alpha=0.85, zorder=zorder_base + 1))
        # Top fold lines
        ax.plot([c - w/2, c + w/2], [r - h/2, r - h/2],
                color="#5e3a14", lw=0.7, zorder=zorder_base + 2)
        ax.plot([c - w/2, c + w/2], [r + h/2, r + h/2],
                color="#5e3a14", lw=0.7, zorder=zorder_base + 2)

    @staticmethod
    def _draw_charger(ax, r, c):
        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.32, r - 0.32), 0.64, 0.64,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc="#f0f0f0", ec="#888888", lw=0.8, zorder=4))
        bolt = [
            (c - 0.04, r - 0.20), (c - 0.16, r + 0.04),
            (c - 0.04, r + 0.04), (c - 0.10, r + 0.20),
            (c + 0.10, r - 0.04), (c - 0.02, r - 0.04),
            (c + 0.04, r - 0.20),
        ]
        ax.add_patch(mpatches.Polygon(
            bolt, closed=True, fc="#ffce20", ec="#b08a10",
            lw=0.7, zorder=5))

    @staticmethod
    def _draw_robot(ax, pos, body_color, facing=3):
        pr, pc = pos
        # Drop shadow
        ax.add_patch(mpatches.Ellipse(
            (pc, pr + 0.30), 0.50, 0.10,
            fc="#000000", alpha=0.45, zorder=7))
        # Antenna (line + red bulb)
        ax.plot([pc, pc], [pr - 0.18, pr - 0.36],
                color="#bbbbbb", lw=1.4, zorder=8)
        ax.add_patch(plt.Circle(
            (pc, pr - 0.40), 0.05,
            fc="#ff3030", ec="#ffaaaa", lw=0.6, zorder=10))
        # Body (rounded rect)
        ax.add_patch(mpatches.FancyBboxPatch(
            (pc - 0.30, pr - 0.20), 0.60, 0.42,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            fc=body_color, ec="#ffffff", lw=1.5, zorder=8))
        # Face panel (slightly darker rectangle in body)
        ax.add_patch(mpatches.FancyBboxPatch(
            (pc - 0.22, pr - 0.13), 0.44, 0.24,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            fc="#10202c", ec="none", zorder=9))
        # Eyes
        eye_dx = 0.12 if facing != 2 else -0.12
        for sgn in (-1, 1):
            ex = pc + sgn * 0.10
            ey = pr - 0.02
            ax.add_patch(plt.Circle((ex, ey), 0.055,
                                     fc="white", ec="none", zorder=10))
            # Pupil — drift toward facing direction
            ax.add_patch(plt.Circle(
                (ex + (eye_dx * 0.12), ey), 0.025,
                fc="#101020", ec="none", zorder=11))
        # Mouth (small smile arc)
        ax.add_patch(mpatches.Arc(
            (pc, pr + 0.07), 0.12, 0.06,
            theta1=0, theta2=180,
            color="#aabbcc", lw=1.2, zorder=11))
        # Wheels (small dark rounded rects under body)
        for sgn in (-1, 1):
            ax.add_patch(mpatches.FancyBboxPatch(
                (pc + sgn * 0.20 - 0.07, pr + 0.20), 0.14, 0.06,
                boxstyle="round,pad=0.005,rounding_size=0.02",
                fc="#202028", ec="#404048", lw=0.5, zorder=8))

    @staticmethod
    def _draw_check(ax, r, c, alpha):
        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.40, r - 0.40), 0.80, 0.80,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            fc="#f4f4f4", ec="#30b048", lw=1.4,
            alpha=alpha, zorder=6))
        # ✓ as a polyline
        ax.plot([c - 0.20, c - 0.05, c + 0.22],
                [r + 0.02, r + 0.20, r - 0.18],
                color="#27a040", lw=4.0, alpha=alpha, zorder=7,
                solid_capstyle="round", solid_joinstyle="round")

    # ─── single panel ───────────────────────────────────────
    def _panel(self, i, env, agent, pos):
        """Two-tier drawing: static layer (floor/walls/charger/drop-off) only
        redraws when the layout changes; dynamic layer (robot/parcel/popup/
        trail/header text) is removed and replaced every frame."""
        ax = self.gax[i]
        H, W = env.cfg.grid_size
        walls    = frozenset(env._walls)
        chargers = frozenset(env.battery_stations)
        goal     = (env.delivery_queue[0].destination
                    if env.delivery_queue else None)
        state_key = (walls, chargers, goal)

        if state_key != self._last_state[i]:
            ax.clear()
            ax.set_facecolor("#0f1015")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            self._draw_floor_fast(ax, H, W)
            if goal is not None:
                self._draw_dropoff(ax, goal[0], goal[1])
            for (r, c) in walls:
                if r == 0 or r == H - 1 or c == 0 or c == W - 1: continue
                self._draw_crate(ax, r, c)
            for (r, c) in chargers:
                self._draw_charger(ax, r, c)
            ax.set_xlim(-.5, W - .5)
            ax.set_ylim(H - .5, -.5)
            ax.set_aspect("equal")
            self._last_state[i] = state_key
            self._dyn[i] = []
        else:
            for a in self._dyn[i]:
                try: a.remove()
                except Exception: pass
            self._dyn[i] = []

        da = self._dyn[i]

        # Traffic overlays (yellow/orange/red translucent tiles)
        traffic = env.traffic.get_grid()
        TRAFFIC_TINT = ["", "#ffd02055", "#ff882080", "#ff2828b0"]
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if (r, c) in walls: continue
                lvl = int(traffic[r, c])
                if lvl == 0: continue
                tile = mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1.0, 1.0,
                    fc=TRAFFIC_TINT[lvl], ec="none", zorder=2)
                ax.add_patch(tile); da.append(tile)
                if lvl >= 2:
                    # Small "!" warning glyph for medium/heavy traffic
                    t = ax.text(c, r, "!", ha="center", va="center",
                                fontsize=11 if lvl == 3 else 9,
                                color="white", fontweight="bold",
                                fontfamily="monospace", alpha=0.85,
                                zorder=3)
                    da.append(t)

        # Persistent ✓ overlay (fades after delivery)
        if self.checks[i] is not None:
            cr, cc, frames = self.checks[i]
            if frames > 0:
                alpha = max(0.0, frames / 30.0)
                p = mpatches.FancyBboxPatch(
                    (cc - 0.40, cr - 0.40), 0.80, 0.80,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    fc="#f4f4f4", ec="#30b048", lw=1.4,
                    alpha=alpha, zorder=6)
                ax.add_patch(p); da.append(p)
                line = ax.plot(
                    [cc - 0.20, cc - 0.05, cc + 0.22],
                    [cr + 0.02, cr + 0.20, cr - 0.18],
                    color="#27a040", lw=4.0, alpha=alpha, zorder=7,
                    solid_capstyle="round", solid_joinstyle="round")[0]
                da.append(line)
                self.checks[i][2] = frames - 1
            else:
                self.checks[i] = None

        # Robot sprite
        pr, pc = pos
        body_color = BOT_COLS[i]
        facing = self.facing[i]
        sh = mpatches.Ellipse((pc, pr + 0.30), 0.50, 0.10,
                              fc="#000000", alpha=0.45, zorder=7)
        ax.add_patch(sh); da.append(sh)
        ant = ax.plot([pc, pc], [pr - 0.18, pr - 0.36],
                      color="#bbbbbb", lw=1.4, zorder=8)[0]
        da.append(ant)
        bulb = plt.Circle((pc, pr - 0.40), 0.05,
                          fc="#ff3030", ec="#ffaaaa", lw=0.6, zorder=10)
        ax.add_patch(bulb); da.append(bulb)
        body = mpatches.FancyBboxPatch(
            (pc - 0.30, pr - 0.20), 0.60, 0.42,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            fc=body_color, ec="#ffffff", lw=1.5, zorder=8)
        ax.add_patch(body); da.append(body)
        face = mpatches.FancyBboxPatch(
            (pc - 0.22, pr - 0.13), 0.44, 0.24,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            fc="#10202c", ec="none", zorder=9)
        ax.add_patch(face); da.append(face)
        eye_dx = 0.12 if facing != 2 else -0.12
        for sgn in (-1, 1):
            ex = pc + sgn * 0.10
            ey = pr - 0.02
            e1 = plt.Circle((ex, ey), 0.055, fc="white",
                            ec="none", zorder=10)
            ax.add_patch(e1); da.append(e1)
            e2 = plt.Circle((ex + eye_dx * 0.12, ey), 0.025,
                            fc="#101020", ec="none", zorder=11)
            ax.add_patch(e2); da.append(e2)
        mouth = mpatches.Arc((pc, pr + 0.07), 0.12, 0.06,
                             theta1=0, theta2=180,
                             color="#aabbcc", lw=1.2, zorder=11)
        ax.add_patch(mouth); da.append(mouth)
        for sgn in (-1, 1):
            w = mpatches.FancyBboxPatch(
                (pc + sgn * 0.20 - 0.07, pr + 0.20), 0.14, 0.06,
                boxstyle="round,pad=0.005,rounding_size=0.02",
                fc="#202028", ec="#404048", lw=0.5, zorder=8)
            ax.add_patch(w); da.append(w)

        # Parcel on the robot while carrying
        if env.completed_deliveries == 0:
            scale = 0.55
            pw, ph = 0.62 * scale, 0.46 * scale
            ppr = pr - 0.34
            box = mpatches.FancyBboxPatch(
                (pc - pw/2, ppr - ph/2), pw, ph,
                boxstyle="round,pad=0.01,rounding_size=0.03",
                fc="#c08550", ec="#704018", lw=1.0, zorder=12)
            ax.add_patch(box); da.append(box)
            tape_h = mpatches.Rectangle(
                (pc - pw/2, ppr - 0.06 * scale), pw, 0.12 * scale,
                fc="#9a6028", ec="none", alpha=0.85, zorder=13)
            ax.add_patch(tape_h); da.append(tape_h)
            tape_v = mpatches.Rectangle(
                (pc - 0.05 * scale, ppr - ph/2), 0.10 * scale, ph,
                fc="#9a6028", ec="none", alpha=0.85, zorder=13)
            ax.add_patch(tape_v); da.append(tape_v)

        # Floating "+100" popup
        if self.popups[i] is not None:
            txt, frames, (gx, gy) = self.popups[i]
            if frames > 0:
                age = 24 - frames
                rise = age * 0.04
                alpha = max(0.0, frames / 24.0)
                t = ax.text(gx, gy - 0.55 - rise, txt,
                            ha="center", va="bottom",
                            fontsize=14 + min(5, age // 2),
                            color="#ffe060", fontweight="bold",
                            fontfamily="monospace",
                            alpha=alpha, zorder=15)
                da.append(t)
                self.popups[i][1] = frames - 1
            else:
                self.popups[i] = None

        # Delivery flash overlay
        if self.flashes[i] > 0:
            fl = mpatches.Rectangle(
                (-.5, -.5), W, H, fc="#00ff6612",
                ec="#00ee66", lw=3, zorder=14)
            ax.add_patch(fl); da.append(fl)
            self.flashes[i] -= 1

        # Header text (live)
        rank = getattr(self, "_cur_ranks", [None, None, None])[i]
        rank_txt, rank_col = "", "#aaccee"
        if rank == 1:   rank_txt, rank_col = "RANK 1  ★", "#ffe060"
        elif rank == 2: rank_txt, rank_col = "RANK 2",    "#cccccc"
        elif rank == 3: rank_txt, rank_col = "RANK 3",    "#bb8866"

        da.append(ax.text(.50, 1.015, LABELS[i],
                          color="white", fontsize=11, fontweight="bold",
                          ha="center", va="bottom", fontfamily="monospace",
                          transform=ax.transAxes, zorder=20))
        da.append(ax.text(.02, 1.015, f"Ep {agent.episode}",
                          color="#aaccee", fontsize=8,
                          ha="left", va="bottom", fontfamily="monospace",
                          transform=ax.transAxes, zorder=20))
        if rank_txt:
            da.append(ax.text(.32, 1.015, rank_txt,
                              color=rank_col, fontsize=8.5, fontweight="bold",
                              ha="left", va="bottom", fontfamily="monospace",
                              transform=ax.transAxes, zorder=20))
        da.append(ax.text(.98, 1.015,
                          f"Score {self.cur_score[i]:>+6.0f}   "
                          f"Best {agent.best:>+6.0f}",
                          color="#aaccee", fontsize=8, ha="right", va="bottom",
                          fontfamily="monospace", transform=ax.transAxes,
                          zorder=20))

    # ─── HUD strip ──────────────────────────────────────────
    def _hud(self, i, env, agent):
        ax = self.hax[i]; ax.clear()
        ax.set_facecolor("#13141c"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(HDR_COLS[i]); sp.set_linewidth(1.8)

        def T(x, y, s, c="white", sz=9, bold=False, align="left"):
            ax.text(x, y, s, color=c, fontsize=sz, va="center", ha=align,
                    fontfamily="monospace",
                    fontweight="bold" if bold else "normal",
                    transform=ax.transAxes)

        T(.02, .80, "EPSILON", "#5566aa", 7.5, bold=True)
        T(.02, .30, "0.400", "#50c0ff", 11, bold=True)
        dcol = "#40ee40" if env.completed_deliveries > 0 else "white"
        T(.22, .80, "PKG", "#5566aa", 7.5, bold=True)
        T(.22, .30, f"{env.completed_deliveries}/{env.cfg.num_deliveries}",
          dcol, 11, bold=True)
        T(.38, .80, "STEPS", "#5566aa", 7.5, bold=True)
        T(.38, .30, f"{env.step_count}/{env.cfg.max_steps}",
          "#cccc40", 9, bold=True)
        note = {"Q-Learning": "off-policy", "SARSA": "on-policy ", "DQN": "neural net"}
        T(.97, .25, note[LABELS[i]], "#445588", 7.5, align="right")

        exploit = max(0., 1. - agent.epsilon)
        ax.add_patch(mpatches.Rectangle((.58, .18), .38, .36,
                                        fc="#101030", ec="#333366",
                                        lw=1, transform=ax.transAxes))
        if exploit > .002:
            ax.add_patch(mpatches.Rectangle((.58, .18), .38 * exploit, .36,
                                            fc=HDR_COLS[i], ec="none",
                                            transform=ax.transAxes))
        T(.58, .80, f"EXPLOIT  {exploit*100:.0f}%", "#5566aa", 7.5, bold=True)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*70}")
    print(f"  Q-Learning vs SARSA vs DQN  —  Warehouse Delivery Robot")
    print(f"  Floor {ENV_CFG.grid_size}   Crates {ENV_CFG.wall_density:.0%}   "
          f"Train seeds {len(TRAIN_SEEDS)}   Demo seeds {len(DEMO_SEEDS)} (UNSEEN)")
    print(f"{'='*70}\n")

    print(f"  training agents on TRAIN_SEEDS:")
    ql    = QLAgent()
    sarsa = SARSAAgent()
    pretrain_tabular(ql,    "Q-Learning ", TABULAR_PRETRAIN_EPS, TRAIN_SEEDS)
    pretrain_tabular(sarsa, "SARSA      ", TABULAR_PRETRAIN_EPS, TRAIN_SEEDS)
    dqn = prepare_dqn(TRAIN_SEEDS)
    ql.episode = 0; ql.best = 0
    sarsa.episode = 0; sarsa.best = 0

    print(f"\n  Starting live demo on UNSEEN seeds. Ctrl+C to exit.\n")

    envs   = [make_env(DEMO_SEEDS[0]) for _ in range(3)]
    agents = [ql, sarsa, dqn]
    rend   = CompareRenderer()

    obs_list = [env.reset(seed=DEMO_SEEDS[0])[0] for env in envs]
    for i in range(3):
        rend.append_trail(i, (float(envs[i].agent_pos[0]),
                               float(envs[i].agent_pos[1])))
    ep_idx     = [0, 0, 0]
    ep_raw     = [0.0, 0.0, 0.0]
    ep_steps   = [0, 0, 0]
    dcache     = [{}, {}, {}]
    pos_hist   = [deque(maxlen=6) for _ in range(3)]
    forced_path= [[], [], []]
    overrides  = [0, 0, 0]
    t0         = time.time()

    print(f"  {'Algo':<12} {'Ep':>3}  {'ε':>5}  {'Result':<10}  "
          f"{'Score':>7}  {'Steps':>5}  {'Suc%':>5}  {'Nav':>4}")
    print(f"  {'-'*68}")

    try:
        while True:
            prev_positions = [(float(env.agent_pos[0]), float(env.agent_pos[1]))
                              for env in envs]
            actions = []
            for i in range(3):
                env, agent = envs[i], agents[i]
                # Stuck detection: if 6 recent positions span ≤2 unique cells,
                # the policy is oscillating — bail out with a BFS path.
                pos_hist[i].append(env.agent_pos)
                stuck = (len(pos_hist[i]) >= 6 and len(set(pos_hist[i])) <= 2)
                if forced_path[i]:
                    a = forced_path[i].pop(0)
                elif stuck:
                    goal_now = (env.delivery_queue[0].destination
                                if env.delivery_queue else env.agent_pos)
                    bp = bfs_path(env.agent_pos, goal_now,
                                  env._walls, env.cfg.grid_size)
                    if bp:
                        forced_path[i] = bp[:8]
                        a = forced_path[i].pop(0) if forced_path[i] else 4
                        overrides[i] += 1
                    else:
                        a = int(np.random.randint(4))
                    pos_hist[i].clear()
                else:
                    a = (agent.act(obs_list[i]) if i == 2
                         else agent.act(disc_state(env)))
                    if a == 4:           # never idle during demo
                        a = int(np.random.randint(4))
                actions.append(a)
                rend.set_last_action(i, a)

            new_positions, results = [], []
            for i in range(3):
                env, agent = envs[i], agents[i]
                a = actions[i]
                prev_pos = (env.agent_pos[0], env.agent_pos[1])
                goal = (env.delivery_queue[0].destination
                        if env.delivery_queue else prev_pos)
                next_obs, raw_r, term, trunc, info = env.step(a)
                curr_pos = env.agent_pos
                done = term or trunc

                ep_raw[i]   += raw_r
                ep_steps[i] += 1
                rend.cur_score[i] = ep_raw[i]

                r = raw_r + shaped(prev_pos, curr_pos, goal,
                                   env._walls, env.cfg.grid_size, dcache[i],
                                   traffic_grid=env.traffic.get_grid())
                if i == 2:
                    agent.store_and_learn(obs_list[i], a, r, next_obs, done)
                else:
                    goal_post = (env.delivery_queue[0].destination
                                 if env.delivery_queue else goal)
                    prev_s = (prev_pos[0], prev_pos[1], goal[0], goal[1])
                    next_s = (curr_pos[0], curr_pos[1], goal_post[0], goal_post[1])
                    agent.learn(prev_s, a, r, next_s, done)

                if done and info["deliveries_remaining"] == 0:
                    rend.notify_delivery(i, (goal[1], goal[0]), goal)

                obs_list[i] = next_obs
                new_positions.append((float(curr_pos[0]), float(curr_pos[1])))
                results.append((done, info))

            for i in range(3):
                rend.append_trail(i, new_positions[i])

            for f in range(1, SMOOTH_FRAMES + 1):
                t = f / SMOOTH_FRAMES
                interp = []
                for i in range(3):
                    pr0, pc0 = prev_positions[i]
                    pr1, pc1 = new_positions[i]
                    interp.append((pr0 + t*(pr1-pr0), pc0 + t*(pc1-pc0)))
                rend.render(envs, agents, interp_pos=interp,
                            full=(f == SMOOTH_FRAMES))
                time.sleep(RENDER_PAUSE)

            for i in range(3):
                done, info = results[i]
                if not done: continue
                success = info["deliveries_remaining"] == 0
                rend.record_episode(i, success, ep_raw[i], ep_steps[i])
                t_now  = time.time() - t0
                result = "DELIVERED " if success else "timed out "
                print(f"  {LABELS[i]:<12} {agents[i].episode+1:>3}  "
                      f"{agents[i].epsilon:>5.3f}  {result}  "
                      f"{ep_raw[i]:>+7.1f}  {ep_steps[i]:>5}  "
                      f"{np.mean(rend.suc_hist[i])*100:>4.0f}%  "
                      f"{overrides[i]:>4}")
                agents[i].on_done(ep_raw[i])
                ep_raw[i]   = 0.0
                ep_steps[i] = 0
                rend.cur_score[i] = 0.0
                ep_idx[i]   = (ep_idx[i] + 1) % len(DEMO_SEEDS)
                obs_list[i], _ = envs[i].reset(seed=DEMO_SEEDS[ep_idx[i]])
                rend.reset_trail(i)
                rend.append_trail(i, (float(envs[i].agent_pos[0]),
                                      float(envs[i].agent_pos[1])))
                dcache[i].clear()
                pos_hist[i].clear()
                forced_path[i] = []
                overrides[i] = 0

    except KeyboardInterrupt:
        pass

    plt.ioff(); plt.close("all")
    _print_winner(rend, agents)


def _print_winner(rend, agents):
    print(f"\n{'='*70}")
    print(f"  FINAL  (last 20 eps each)")
    print(f"  {'Algo':<12} {'Episodes':>8}  {'Success':>8}  {'AvgScore':>9}  "
          f"{'AvgSteps':>9}  {'Best':>7}")
    print(f"  {'-'*68}")
    rows = []
    for i in range(3):
        suc = np.mean(rend.suc_hist[i])*100 if rend.suc_hist[i] else 0
        avs = np.mean(rend.scr_hist[i])      if rend.scr_hist[i] else 0
        avp = np.mean(rend.stp_hist[i])      if rend.stp_hist[i] else 0
        rows.append((LABELS[i], agents[i].episode, suc, avs, avp, agents[i].best))
        print(f"  {LABELS[i]:<12} {agents[i].episode:>8}  {suc:>7.0f}%  "
              f"{avs:>+9.1f}  {avp:>9.1f}  {agents[i].best:>+7.0f}")

    ranked = sorted(rows, key=lambda r: -r[3])
    if ranked and ranked[0][1] >= 1:
        winner = ranked[0]
        print(f"\n  ★ WINNER: {winner[0]}  (avg score {winner[3]:+.1f})")
        for r in ranked[1:]:
            gap = winner[3] - r[3]
            print(f"      vs {r[0]:<12}  avg {r[3]:+.1f}   gap {gap:+.1f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
