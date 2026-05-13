# Warehouse Delivery Robot — Q-Learning vs SARSA vs DQN

A side-by-side reinforcement-learning comparison: three agents control three
robots solving the same delivery task in parallel, with a live leaderboard
ranking them by success rate, score, and speed.

---

## 1. Problem statement

A delivery robot operates on a 10×10 warehouse floor. It must:

1. Reach a randomly placed cardboard parcel (the **goal cell**).
2. Avoid wooden **crates** (impassable obstacles).
3. Avoid **traffic zones** (cells with 1–3 levels of congestion that incur
   per-step penalties).
4. Do all of this within `max_steps = 80` time-steps.

The same map is given to all three agents per episode comparison cycle (each
agent owns its own env so they don't physically collide). Demo episodes use
**unseen seeds** that none of the agents trained on, so we are measuring
generalization, not memorization.

---

## 2. Environment (`environment/city_env.py`)

A custom `gymnasium.Env` named `CityDeliveryEnv`.

### State / observation

The agent receives a `Dict` observation:

- **Local FOV**: a `(5, 5, 5)` patch centered on the agent. The five channels
  are: `wall`, `congestion_level`, `delivery_targets`, `battery_stations`,
  `agent`. `fov_radius = 2`.
- **Global vector**: 11-dim `[agent_r/H, agent_c/W, battery/max,
  steps_left/max, next_dest_r/H, next_dest_c/W, next_dest_priority/3,
  deliveries_remaining/num_deliveries, in_rush_hour, …]`.

### Action space

`Discrete(5)`: `0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT, 4 = STAY`.

### Rewards (per env step)

| Event                     | Reward |
|---------------------------|--------|
| Delivery completed        | +100   |
| Delivered within deadline | +50    |
| Step (movement / stay)    |  −0.5  |
| Step on a traffic cell    |  −5    |
| Episode timeout           | −50    |
| Hit a wall (no movement)  | counted as −0.5 step |

### Traffic (`environment/traffic_manager.py`)

Markov-chain dynamic congestion:

- `num_traffic_zones = 4` randomly seeded "zones" radiating outward at
  levels 0–3.
- Each step, a fraction `traffic_transition_prob = 0.05` of cells transition
  to a neighbouring level — congestion shifts on its own.

---

## 3. Three algorithms (one per file)

### 3.1 Q-Learning — `agents/q_learning_agent.py`

Tabular **off-policy** TD(0) control. State key is
`(agent_r, agent_c, goal_r, goal_c)`. Update:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a')  −  Q(s, a) ]
```

`α = 0.30, γ = 0.95`. Linear ε decay from 1.0 → 0.06 over 200 episodes.

### 3.2 SARSA — `agents/sarsa_agent.py`

Tabular **on-policy** TD(0) control. Same state key. Update:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · Q(s', a')  −  Q(s, a) ]
```

with `a'` actually sampled from the ε-greedy policy at `s'` (not the max).
On-policy methods learn the value of the policy *they follow*, so they are
more conservative around penalty regions like traffic.

### 3.3 DQN — `agents/dqn_agent.py` + `models/networks.py`

**Dueling Double-DQN** with a CNN+MLP encoder.

- **Network** (`models/networks.py`): a CNN over the local FOV feeds an MLP,
  concatenated with the global vector, then split into a value head V(s)
  and an advantage head A(s, a). Q(s, a) = V(s) + A(s, a) − mean_a A(s, ·).
- **Replay buffer** (`agents/replay_buffer.py`): uniform circular buffer
  with capacity 8000.
- **Double DQN target**: `argmax` action picked by the online net,
  bootstrap value from the target net — reduces TD overestimation bias.
- **Hyperparameters**: hidden dims `[128, 128]`, lr `1e-3`, γ 0.95,
  batch 32, target update every 150 grad steps, ε 1.0 → 0.06.
- Pretrained for 600 episodes on `TRAIN_SEEDS = [0..30)`, cached to
  `checkpoints/compare_dqn_v6.pt`. Training is one-time (~45 s); subsequent
  launches load instantly.

### Why DQN should beat tabular here

The tabular state `(agent_r, agent_c, goal_r, goal_c)` does **not include
traffic**. Q-Learning and SARSA literally cannot tell a clear cell from a
red gridlock cell — they walk into them and bleed −5 per step. The DQN's
local FOV has a `congestion_level` channel, so the network can learn
"if there's a red cell in front of me, route around it." That is the
qualitative gap the comparison is built to surface.

---

## 4. Reward shaping (`compare_run.py:shaped`)

Sparse delivery rewards alone are too weak to learn from in 30 train seeds.
We add a **potential-based shaped term** on top of the env reward:

```
shaped = 8 · ( BFS(prev, goal) − BFS(curr, goal) )
       − 3        if the agent didn't move (bumped a wall)
       − 4 · lvl  if curr_pos is on a traffic cell of level `lvl`
```

- `8 × ΔBFS` rewards real progress around walls (not Manhattan).
- The traffic-cost term `−4 × level` is what makes detouring around a red
  cell genuinely net-positive: walking through a level-3 cell costs
  `+8 (forward) − 12 (shaping) − 5 (env) = −9`, but a one-step detour costs
  `+0` — so the optimal policy detours.

Shaping is potential-based on geometry, so it does not change the optimal
policy in tabular settings (Ng et al. 1999); it just speeds up learning.

---

## 5. Live demo flow (`compare_run.py:main`)

1. **Pretrain** Q-Learning and SARSA for 280 episodes each on `TRAIN_SEEDS`
   (silent, ~5 s each).
2. **Load or train** the DQN: if `checkpoints/compare_dqn_v6.pt` exists it
   loads in milliseconds, otherwise it trains fresh for 600 episodes.
3. **Demo loop on `DEMO_SEEDS = [50..100)`** — these layouts are *unseen*
   by all three agents during pretrain. Each agent owns its own env; all
   three step in lock-step every iteration.

### Stuck detector + BFS fallback

A safety net for ε-greedy oscillation traps. For each agent we track the
last 6 positions; if the agent has visited ≤ 2 unique cells in that window
it is "stuck". We then compute a BFS path from current pos to goal
(ignoring traffic) and **force the next 8 actions** from that path. The
override count is logged per-episode in the `Nav` column — a cleaner
behavioural metric than raw success because it shows *who needed help*.

In our reference run (15 episodes / agent on unseen seeds):

```
Q-Learning  succ 15/15  avg_score +214.1  avg_steps 16.3  Nav overrides 19
SARSA       succ 15/15  avg_score +215.1  avg_steps 14.3  Nav overrides 18
DQN         succ 15/15  avg_score +219.3  avg_steps 10.5  Nav overrides  8
```

DQN wins on **score** (+5 over SARSA), is **35 % faster** (10.5 vs 14.3
steps), and needs the fallback **less than half as often** (8 vs 18) —
because its FOV-based observation lets it see traffic the tabular agents
can't.

---

## 6. Visualization (`compare_run.py:CompareRenderer`)

Top-down warehouse with sprite-style rendering, three panels side-by-side
plus a live leaderboard strip.

### Performance optimizations

- **Wood floor** is pre-rendered into a numpy RGB array once at startup
  (`_make_floor_img`) and drawn with a single `ax.imshow` per panel
  instead of one `Rectangle` per cell.
- **Two-tier rendering**: floor / crates / charger / drop-off pad are
  drawn once when the layout changes (state-key check); only the dynamic
  artists (robot, parcel, popup, header text, traffic overlays) are
  removed and replaced each frame. Avoids `ax.clear()` per frame.
- HUD and leaderboard skip redrawing on intermediate interp frames.
- Result: ~100 ms per env step, ≈ 5 cells/sec gliding animation.

### Sprites

- **Robot**: body, face panel, two eyes with pupils that drift toward the
  facing direction, smile arc, antenna with red bulb, two wheels. Body
  colour identifies the algorithm (blue = Q-Learning, red = SARSA,
  green = DQN).
- **Parcel**: cardboard box with horizontal + vertical tape stripes.
  Drawn on top of the robot while `env.completed_deliveries == 0`.
- **Goal**: green delivery pad with concentric white bullseye target.
  After delivery, a fading white card with a green ✓ overlays the
  destination tile.
- **Crate**: dark wooden box with two horizontal slats.
- **Charger**: white panel with a yellow lightning-bolt polygon.
- **Traffic**: translucent yellow / orange / red rectangles per affected
  cell, plus a white "!" glyph on level-2 and level-3 tiles.

### Smooth animation

`SMOOTH_FRAMES = 2` interpolated frames per env step — the robot slides
between cells rather than teleporting. `RENDER_PAUSE = 0.005` keeps
animation snappy without saturating the CPU.

### Leaderboard (top strip)

For each algorithm: live `Success %`, `Avg Score`, and `Speed` bars
(normalized against the current leader so the gap is visible), plus
`EP n`, `✓ deliveries`, and a `★ LEADER` badge once at least 2 episodes
have completed for that agent. Each panel header also carries a live
`RANK 1 ★ / RANK 2 / RANK 3` badge.

---

## 7. File structure (after cleanup)

```
RL_project/
├── compare_run.py               # demo entry point
├── abstract.md                  # this file
├── requirements.txt
│
├── agents/
│   ├── q_learning_agent.py      # tabular off-policy
│   ├── sarsa_agent.py           # tabular on-policy
│   ├── dqn_agent.py             # dueling DDQN wrapper + training
│   └── replay_buffer.py
│
├── environment/
│   ├── city_env.py              # CityDeliveryEnv (gym.Env)
│   └── traffic_manager.py       # Markov-chain congestion
│
├── models/
│   └── networks.py              # CNN + MLP, dueling head
│
├── utils/
│   └── config.py                # EnvConfig, DQNConfig dataclasses
│
└── checkpoints/
    └── compare_dqn_v6.pt        # ~600-episode pretrained DQN
```

Run the demo:

```
python3 compare_run.py
```

---

## 8. Cross-questions you might be asked

**Q. Why three algorithms? Why not just DQN?**
> The point of the project is *comparison*, not just training. The three
> algorithms are textbook representatives of three families: off-policy
> tabular (Q-Learning), on-policy tabular (SARSA), and function-approximation
> deep RL (DQN). Showing them side-by-side on the same task makes the
> trade-offs concrete.

**Q. What is the difference between Q-Learning and SARSA?**
> Both are TD(0) control methods. Q-Learning is **off-policy**: its TD
> target uses `max_a' Q(s', a')`, i.e. the value of the *greedy* next
> action regardless of what the agent actually does. SARSA is
> **on-policy**: its target uses `Q(s', a')` where `a'` is the action
> actually sampled from the ε-greedy policy at `s'`. On-policy SARSA learns
> the value of the policy it follows (including its exploration), so it is
> more risk-averse near penalty regions. Off-policy Q-Learning converges
> to the optimal `Q*` regardless of the behaviour policy.

**Q. Why is DQN dueling double DQN, not vanilla DQN?**
> Two reasons. **Double DQN** decouples action selection from value
> estimation in the bootstrap target — the online net picks `argmax_a`,
> the target net evaluates it — which mitigates the systematic
> overestimation bias of `max_a Q_target(s', a)` (van Hasselt 2015).
> **Dueling** factors `Q(s, a) = V(s) + A(s, a) − mean_a A(s, ·)` so the
> network can estimate state values cheaply for states where the action
> doesn't matter (most of an empty floor) and only spend advantage capacity
> where it does (Wang et al. 2016).

**Q. What is the role of the replay buffer?**
> Decorrelates consecutive transitions by sampling random mini-batches
> from a fixed-size circular buffer. Required for stable gradient updates
> in DQN — directly applying SGD to highly correlated trajectory data
> diverges. Capacity 8000 here.

**Q. What is the target network for?**
> The TD target depends on the current Q-network, so the target moves with
> every gradient step — chasing your own tail. The target network is a
> *frozen copy* updated every 150 grad steps; the online net is regressed
> onto its predictions. Stabilizes training (Mnih et al. 2015).

**Q. What is reward shaping and is it cheating?**
> Reward shaping adds an extra signal `F(s, s')` on top of the env reward.
> It is **not cheating** when the shaping is *potential-based*, i.e.
> `F(s, s') = γ · Φ(s') − Φ(s)` for some potential Φ — Ng, Harada, Russell
> (1999) proved this preserves the optimal policy. Our shaping uses
> `Φ(s) = −8 × BFS_distance(s, goal)` plus a one-step penalty for traffic
> cells. The optimal navigation policy is unchanged; we just speed up
> learning by densifying the reward.

**Q. Why does DQN beat the tabular agents on traffic?**
> The tabular state `(ar, ac, gr, gc)` doesn't include traffic. Q-Learning
> and SARSA can't even *see* a red cell — they assign one Q-value per
> `(start, goal)` configuration, averaged over all traffic distributions
> they happened to encounter at that configuration during training. DQN's
> FOV observation has an explicit congestion channel, so the network
> learns a *traffic-conditional* policy. Same map, richer state space.

**Q. Why use disjoint train and demo seeds?**
> To measure generalization, not memorization. If we trained and
> demo-ed on the same seeds, tabular agents would just look up the
> Q-value for that exact `(start, goal)` and look unreasonably good. With
> `TRAIN_SEEDS = [0..30)` and `DEMO_SEEDS = [50..100)`, the demo layouts
> are unseen — the only way to perform well is to *generalize*, which is
> where DQN's parametric function approximation helps.

**Q. What is ε-greedy and why anneal ε?**
> ε-greedy picks a uniform random action with probability ε, otherwise
> the greedy `argmax_a Q(s, a)`. We start at ε = 1.0 (pure exploration)
> and linearly decay to 0.06 (mostly greedy) — explore early to fill the
> Q-table / replay buffer with diverse states, exploit later when the
> value estimates are reliable.

**Q. What happens if the agent gets stuck?**
> Despite training, ε-greedy at low ε can land in oscillation traps where
> two cells have nearly-identical Q-values. We detect this in the demo
> loop: if the last 6 positions span ≤ 2 unique cells, we compute a BFS
> path to the goal and force the next 8 actions from it. The override
> count is logged per-episode (`Nav` column) — DQN needs it least often,
> which is itself a measure of policy quality.

**Q. What is the role of the BFS in shaping AND in the stuck detector?**
> Two different uses. In **shaping** we use BFS distance as the potential
> Φ, giving a per-step gradient toward the goal. In the **stuck detector**
> we use BFS to *plan* a sequence of actions when the learned policy is
> oscillating. Both rely on the same reachability graph (cells connected
> by walls), but for very different purposes.

**Q. How does the dueling head improve learning?**
> Many states have similar values regardless of action (e.g. an empty
> patch of floor — every direction is fine, only "toward goal" matters).
> A vanilla DQN has to learn that lateral indifference five times, once
> per action. Dueling factors `Q(s, a) = V(s) + (A(s, a) − mean_a A)`,
> so the value head V(s) handles "how good is being here at all?"
> independently of action — better sample efficiency.

**Q. What does the FOV observation give that the global vector doesn't?**
> The 11-dim global vector tells the agent *where it is and where the
> goal is* — but not what is in between. The 5×5 FOV gives a local
> spatial view: nearby walls, nearby chargers, **nearby traffic
> intensity**. Without the FOV, the network can't make traffic-aware
> decisions because it has no input feature for traffic.

**Q. Why limit max_steps?**
> Without a cap, a stuck agent would run forever and never terminate, so
> we couldn't compute a return. `max_steps = 80` is roughly 4× the
> optimal path length for a 10×10 grid, generous enough that any
> reasonable policy delivers, tight enough that pathological loops fail
> the episode and we can score that as a timeout (−50).

**Q. What is the training cost?**
> Tabular pretrain: ~5 seconds each. DQN pretrain: ~45 seconds (CPU,
> 600 episodes × ~30 steps × learn-every-4 = ~4500 gradient updates on a
> 128×128 MLP head). Cached after first run; subsequent launches are
> instant.

**Q. Could you scale this to a real warehouse?**
> The env supports difficulty 0–3 (10×10 → 15×15, more deliveries, rush-
> hour events, dynamic incidents). The CNN/MLP architecture scales by
> swapping the FOV size and rebuilding. For a real warehouse you'd want
> richer observations (LiDAR / camera), continuous actions (velocity),
> multi-agent coordination, and probably PPO or SAC instead of DQN —
> but the comparison framework here is the same.

---

## 9. 90-second technical pitch

> Hi — this project compares three reinforcement-learning algorithms on
> the same warehouse delivery task: tabular Q-Learning, tabular SARSA,
> and a Dueling Double DQN. A robot navigates a 10-by-10 floor avoiding
> wooden crates and dynamically shifting traffic zones to deliver a
> parcel to a green drop-off pad.
>
> The environment is a custom Gym env with a five-channel local FOV
> observation plus an 11-dim global vector, a discrete five-action space
> — UP, DOWN, LEFT, RIGHT, STAY — and a reward function that combines
> a +100 delivery bonus, a −0.5 step cost, and a −5 traffic penalty.
> I add a potential-based BFS-distance shaping plus a per-cell traffic
> cost on top, so the optimal policy is preserved but learning is faster.
>
> The crucial design choice is asymmetric observation. The two tabular
> agents see only `(agent position, goal position)` — they're literally
> blind to traffic. The DQN sees a 5-by-5 patch with a congestion
> channel, so it can learn a traffic-conditional policy. Pretrain runs
> on 30 seeds; demo runs on 50 *unseen* seeds, so we're measuring
> generalization, not memorization.
>
> A dueling head splits Q into V(s) + A(s, a), which is more
> sample-efficient when many states are action-indifferent. Double DQN
> decouples action selection from bootstrap evaluation to fight max-bias.
> A stuck-detector with BFS fallback catches ε-greedy oscillation traps
> in the demo so all three agents always deliver, and the override count
> per episode becomes a clean behavioural metric: lower is better.
>
> On 15 unseen-seed episodes, DQN delivers in 10.5 average steps versus
> 16.3 for Q-Learning, scores +5 higher than SARSA, and needs the BFS
> fallback less than half as often. The UI runs at about 5 cells per
> second with a live leaderboard, RANK badges per panel, and animated
> sprites — wood-plank floor, cute robot, traffic overlays in red. The
> point of the project isn't just "DQN works"; it's that you can *see*
> why it works, head-to-head, in real time.

---
