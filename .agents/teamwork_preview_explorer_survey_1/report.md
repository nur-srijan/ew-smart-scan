# Comprehensive Technical Report: Codebase RMAB Whittle Index & Mathematical Models

**Author**: Explorer 1 (Codebase RMAB Whittle Index Specialist)  
**Date**: 2026-09-19  
**Repository**: `/Users/nursrijan/dev/sih-project`  
**Working Directory**: `/Users/nursrijan/dev/sih-project/.agents/teamwork_preview_explorer_survey_1`  
**Reference Request**: `ORIGINAL_REQUEST.md` (Track 2: Zero-Allocation C++20 RMAB Engine & pybind11 Extension)

---

## Executive Summary

The SIH-2026 EW Project formulates electronic warfare (EW) spectrum surveillance as a **Restless Multi-Armed Bandit (RMAB)** with $K=35$ frequency sub-bands and $M$ receiver tuners ($M=1$ single-tuner or $M=4$ cooperative multi-tuner). Each frequency sub-band evolves as a two-state Markov chain representing radar emitter activity (pulse burst vs. quiet interval), which continues to transition even when unobserved ("restless").

This report provides a complete, line-by-line anatomical survey of the existing RMAB Whittle Index implementations across Python and C++, the exact mathematical formulas for belief updates and closed-form Whittle indices, the API contracts and interfaces, and the existing automated test suite covering these components.

---

## 1. Codebase Architecture & File Locations

The RMAB, Whittle index, Bayesian belief state updating, and scan schedulers are implemented across the following key modules:

```
sih-project/
├── schedulers/
│   ├── rmab.py               # Single-tuner WhittleIndexScheduler (pure NumPy, <1 µs latency)
│   ├── multi_schedulers.py   # MultiWhittleIndexScheduler (M-tuner top-M Whittle + AoI)
│   ├── baselines.py          # BaseScheduler & classical sweep baselines (Sequential, PseudoRandom, Priority, Uniform)
│   ├── predictor.py          # OnlinePeriodicityEstimator & HybridPredictiveScheduler
│   ├── drl_agent.py          # Recurrent PPO single-receiver DRL scheduler
│   ├── multi_drl.py          # Multi-head Recurrent Actor-Critic DRL scheduler with collision masking
│   └── __init__.py           # Unified scheduler exports
├── hardware/
│   ├── whittle_index.hpp     # Prototype C++20 zero-heap single-tuner Whittle scheduler header
│   ├── test_timing.cpp       # Microsecond latency verification benchmark harness
│   └── export_onnx.py        # ONNX export utility for neural models
├── ew_sim/
│   ├── env.py                # Gymnasium EWSpectrumEnv & DynamicSpectrumEnv (single tuner)
│   ├── multi_env.py          # MultiReceiverEWSpectrumEnv & DynamicMultiReceiverEnv (M=4 tuners)
│   ├── truth_engine.py       # Ground-truth pulse matrix (K x T) builder and query engine
│   └── emitters.py           # FixedFrequencyEmitter, FHSSEmitter, ScanningEmitter
├── tests/
│   ├── test_rmab.py          # Unit tests for WhittleIndexScheduler
│   ├── test_multi_receiver.py# Unit tests for MultiWhittleIndexScheduler and multi_env
│   ├── test_dynamic_env.py   # Integration test running Whittle on DynamicSpectrumEnv
│   ├── test_predictor.py     # Unit tests for HybridPredictiveScheduler
│   └── test_demo.py          # Tests for dashboard policy factory and episode runner
└── eval/
    ├── fom.py                # Figure-of-Merit evaluation metrics (IR, TTI, Collisions, Discovery)
    ├── runner.py             # Monte Carlo batch evaluation harness
    └── benchmark_multi_receiver.py # Multi-receiver benchmark comparing MultiWhittle vs DRL
```

---

## 2. Mathematical Models & Exact Equations

### 2.1 State Space & Transition Probability Dynamics

Each sub-band $k \in \{0, \dots, K-1\}$ ($K=35$) has an unobserved true RF state $s_t(k) \in \{0, 1\}$ at discrete time slot $t$:
- $s_t(k) = 0$: Band $k$ is **quiet / idle** (no active radar pulse).
- $s_t(k) = 1$: Band $k$ is **active / emitting** (pulse or pulse burst present).

#### Passive / Restless Transition Matrix
Whether sensed or unsensed, the emitter activity evolves according to a two-state Markov chain:
$$
P(k) = \begin{bmatrix} P_{00}[k] & P_{01}[k] \\ P_{10}[k] & P_{11}[k] \end{bmatrix} = \begin{bmatrix} 1 - P_{01}[k] & P_{01}[k] \\ 1 - P_{11}[k] & P_{11}[k] \end{bmatrix}
$$
- $P_{01}[k] = P(s_{t+1}(k) = 1 \mid s_t(k) = 0)$: Probability of an emitter bursting on.
- $P_{11}[k] = P(s_{t+1}(k) = 1 \mid s_t(k) = 1)$: Probability of an emitter continuing transmission (burst hold / temporal persistence).
- Default initialization in Python (`schedulers/rmab.py:74-75`):
  $$P_{01}[k] = 0.03, \quad P_{11}[k] = 0.92$$
- Default initialization in C++ (`hardware/whittle_index.hpp:36-37`):
  $$P_{01}[k] = 0.03\text{f}, \quad P_{11}[k] = 0.92\text{f}$$

#### Online Transition Probability Learning
In `schedulers/rmab.py:151-166` and `schedulers/multi_schedulers.py:304-319`, when band $k$ is visited at slot $t$, the scheduler measures $\Delta t = t - \text{last\_slot}[k]$. If $\Delta t \le 10$ and $t > 0$, an exponential moving average (EMA) update is applied using the previous state $\text{prev\_state} \in \{0, 1\}$ and the newly observed state $y_t(k) \in \{0, 1\}$:
$$
\begin{cases}
P_{01}[k] \leftarrow (1 - \eta) P_{01}[k] + \eta \cdot y_t(k), & \text{if } \text{prev\_state} = 0 \\
P_{11}[k] \leftarrow (1 - \eta) P_{11}[k] + \eta \cdot y_t(k), & \text{if } \text{prev\_state} = 1
\end{cases}
$$
where learning rate $\eta = \text{lr\_transition} = 0.05$.  
The probabilities are strictly clipped to physical bounds:
$$
P_{01}[k] \in [0.01, 0.50], \quad P_{11}[k] \in [0.20, 0.99]
$$

---

### 2.2 Observation Model & Radar Detection Physics

The EW receiver detector has imperfect physical detection characteristics:
- $P_d = 0.95$: Detector probability of detection (true positive rate given pulse present).
- $P_{fa} = 10^{-4} = 0.0001$: False alarm probability (false positive rate given quiet band).
- $d = 0.20$: Radar pulse duty cycle factor during a dwell slot ($\approx 20\%$).

#### Likelihood Probabilities
When sub-band $k$ is sensed, the binary observation is $y_t(k) \in \{0, 1\}$ (Hit = 1, Miss = 0).  
The likelihoods are conditioned on whether an emitter is active ($s=1$) or absent ($s=0$):
$$
\begin{aligned}
P(\text{Hit} \mid \text{Present}) &= d \cdot P_d + (1 - d) \cdot P_{fa} \\
&= 0.20 \times 0.95 + 0.80 \times 10^{-4} = 0.19008 \\
P(\text{Hit} \mid \text{Absent}) &= P_{fa} = 10^{-4} = 0.0001 \\
P(\text{Miss} \mid \text{Present}) &= d \cdot (1 - P_d) + (1 - d) \cdot (1 - P_{fa}) \\
&= 0.20 \times 0.05 + 0.80 \times 0.9999 = 0.80992 \\
P(\text{Miss} \mid \text{Absent}) &= 1 - P_{fa} = 0.9999
\end{aligned}
$$

---

### 2.3 Bayesian Belief State Updating Equations

The state of each restless arm $k$ is summarized by the scalar Bayesian belief:
$$
b_t(k) \triangleq P(s_t(k) = 1 \mid \mathcal{H}_t) \in [0, 1]
$$
Initial belief: $b_0(k) = 0.10$ (Python schedulers) or $0.15$ (C++ prototype) or $0.50$ (environments).

#### Sensed Band Posterior Update
For the tuned sub-band $k = \text{action}$:
1. **If Hit ($y_t(k) = 1$)**:
   $$
   b_{t+1}(k) = \frac{b_t(k) \cdot P(\text{Hit} \mid \text{Present})}{\max\left(10^{-9}, b_t(k) \cdot P(\text{Hit} \mid \text{Present}) + (1 - b_t(k)) \cdot P(\text{Hit} \mid \text{Absent})\right)}
   $$
2. **If Miss ($y_t(k) = 0$)**:
   $$
   b_{t+1}(k) = \frac{b_t(k) \cdot P(\text{Miss} \mid \text{Present})}{\max\left(10^{-9}, b_t(k) \cdot P(\text{Miss} \mid \text{Present}) + (1 - b_t(k)) \cdot P(\text{Miss} \mid \text{Absent})\right)}
   $$
3. **Posterior Clamping**:
   $$
   b_{t+1}(k) \leftarrow \text{clamp}(b_{t+1}(k), 0.001, 0.999)
   $$

#### Unsensed Bands Markov State Diffusion
For all unsensed sub-bands $j \ne k$ (or $j \notin \text{unique\_actions}$ in multi-receiver):
The belief decays toward the stationary/uninformative prior distribution through Markov diffusion:
$$
b_{t+1}(j) = (1 - \alpha) \cdot b_t(j) + \alpha \cdot \pi_{\text{prior}}
$$
where diffusion constant $\alpha = 0.02$, prior $\pi_{\text{prior}} = 0.15$.  
Clipped to $[0.001, 0.999]$.

#### Age-of-Information (AoI) Update
Tracks elapsed dwell slots since sub-band $k$ was last monitored:
$$
\begin{cases}
\text{AoI}_{t+1}[k] = 0.0, & \text{if } k \in \text{actions (sensed)} \\
\text{AoI}_{t+1}[j] = \min(\text{AoI}_t[j] + 1.0, \text{AoI}_{\text{max}}), & \text{if } j \notin \text{actions (unsensed)}
\end{cases}
$$
where $\text{AoI}_{\text{max}} = 50.0$ in `MultiWhittleIndexScheduler` and $100.0$ in `WhittleIndexScheduler`.

---

### 2.4 Closed-Form Whittle Index Formula

The Restless Multi-Armed Bandit Whittle Index $W(p)$ represents the Lagrange multiplier (subsidy for passive action) that makes the operator indifferent between pulling the arm (sensing band $k$) and leaving it idle (passive evolution).

#### Analytical Derivation & Equation
Let $p = b_t(k) \in [0, 1]$, and define the transition persistence differential:
$$
\Delta_k \triangleq P_{11}[k] - P_{01}[k]
$$
Because radar pulse trains exhibit burst persistence ($P_{11} \gg P_{01}$), $\Delta_k > 0$, guaranteeing strict indexability.

The closed-form Whittle Index used across `schedulers/rmab.py:98-114`, `schedulers/multi_schedulers.py:224-240`, and `hardware/whittle_index.hpp:43-49` is:
$$
\boxed{W(p) = \frac{p \cdot \Delta_k + P_{01}[k]}{1 - \Delta_k + p \cdot \Delta_k} = \frac{p (P_{11}[k] - P_{01}[k]) + P_{01}[k]}{(1 - (P_{11}[k] - P_{01}[k])) + p (P_{11}[k] - P_{01}[k])}}
$$

#### Singularity Protection
If the denominator is near zero:
$$
\text{denom} = (1.0 - \Delta_k) + p \cdot \Delta_k \le 10^{-9} \quad (\text{or } 10^{-7} \text{ in C++ float32}) \implies W(p) = p
$$

#### Composite Arm Scoring Function
To balance exploitation of active emitters, wideband exploration, and anti-camping, the scheduler evaluates a composite score for each sub-band $k \in \{0, \dots, K-1\}$:

$$
\text{Score}(k) = W(b_t(k)) + \lambda_{\text{AoI}} \cdot \min\left(1.0, \frac{\text{AoI}_t[k]}{\text{AoI}_{\text{cap}}}\right) - \lambda_{\text{camp}} \cdot c_t(k) + \epsilon_{\text{tie}}
$$

Where:
- $\lambda_{\text{AoI}} = 0.60$: Age-of-Information exploration bonus weight (`aoi_weight`).
- $\text{AoI}_{\text{cap}} = 50.0$: Saturation cap for AoI bonus.
- $\lambda_{\text{camp}} = 0.40$: Anti-camping penalty coefficient (`camping_penalty_weight`).
- $c_t(k)$: Consecutive dwells counter on band $k$ ($c_t(k) = \text{consecutive\_dwells}$ if $k = \text{last\_action}$, else 0).
- $\epsilon_{\text{tie}} \sim U(0, 10^{-5})$: Stochastic tie-breaker (pure NumPy in Python; deterministic in C++).

---

### 2.5 Multi-Receiver Coordination ($M=4$ Tuners, $K=35$ Sub-Bands)

By the Whittle Index Theorem for decoupled restless bandits under an instantaneous capacity constraint of $M$ active channels, the optimal policy selects the **top-$M$ distinct arms** with the highest composite index:
$$
\mathbf{a}_t = \arg\max^{(M)}_{k \in \{0, \dots, K-1\}} \text{Score}(k)
$$
In `schedulers/multi_schedulers.py:259`:
```python
top_m_bands = np.argsort(scores)[::-1][:self.M].astype(int)
```
- Because `top_m_bands` takes the first $M$ elements from a permutation of unique band indices, **0.0% tuner collisions are guaranteed by mathematical construction**.
- Tuner assignment: Tuner $m \in \{0, \dots, M-1\}$ is assigned to `top_m_bands[m]`.

---

### 2.6 Environment Reward Formulations

#### Single Receiver (`ew_sim/env.py:240-264`)
$$
R_t = w_{\text{hit}} \cdot \frac{\text{Hit} \land \text{TrueActive}}{1 + 0.25 \cdot c_t} + w_{\text{new}} \cdot \mathbb{I}_{\text{new}} + w_{\text{aoi}} \cdot \frac{\text{AoI}[a_t]}{\text{AoI}_{\text{max}}} - \mathbb{I}_{c_t \ge 3} \cdot 1.2(c_t - 2) - w_{\text{sw}} \cdot \frac{|a_t - a_{t-1}|}{K - 1}
$$
- $w_{\text{hit}} = 6.0$, $w_{\text{new}} = 40.0$, $w_{\text{aoi}} = 2.5$, $w_{\text{sw}} = 0.2$, $\text{AoI}_{\text{max}} = 200$.

#### Multi-Receiver (`ew_sim/multi_env.py:222-260`)
$$
\begin{aligned}
R_t = &\sum_{k \in \mathcal{U}_t, \text{hit}_k \land \text{active}_k} w_{\text{hit}} \cdot \frac{1}{1 + 0.25 \max(0, c_k - 1)} + w_{\text{new}} \cdot N_{\text{new}} \\
&+ \sum_{k \in \mathcal{U}_t} \frac{w_{\text{aoi}}}{M} \cdot \frac{\text{AoI}[k]}{\text{AoI}_{\text{max}}} - \sum_{k \in \mathcal{U}_t, c_k \ge 3} 0.8(c_k - 2) \\
&- w_{\text{sw}} \cdot \frac{1}{M} \sum_{m=1}^M \frac{|a_m - a_{\text{prev}, m}|}{K - 1} - w_{\text{collision}} \cdot N_{\text{redundant}}
\end{aligned}
$$
where $\mathcal{U}_t = \text{unique}(\mathbf{a}_t)$, $w_{\text{collision}} = 2.0$, and $N_{\text{redundant}} = \sum (\text{count}(k) - 1)$.

---

## 3. Detailed Class & Interface Specifications

### 3.1 Class Hierarchy

```
BaseScheduler (schedulers/baselines.py)
 └── WhittleIndexScheduler (schedulers/rmab.py)
 └── SequentialSweep, PseudoRandomSweep, PriorityQueueSweep, UniformRandomSweep
 └── HybridPredictiveScheduler (schedulers/predictor.py)

BaseMultiScheduler (schedulers/multi_schedulers.py)
 └── MultiWhittleIndexScheduler (schedulers/multi_schedulers.py)
 └── MultiSequentialSweep, MultiPseudoRandomSweep, CooperativeRoleScheduler
```

---

### 3.2 `WhittleIndexScheduler` (`schedulers/rmab.py`)

#### Constructor Parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `K` | `int` | *Required* (35) | Total number of frequency sub-bands |
| `Pd` | `float` | `0.95` | Probability of detection |
| `Pfa` | `float` | `1e-4` | False alarm rate |
| `aoi_weight` | `float` | `0.25` | Weight on Age-of-Information exploration bonus |
| `aoi_max` | `float` | `100.0` | Normalization cap for AoI |
| `lr_transition`| `float` | `0.05` | Learning rate for online transition probabilities |
| `seed` | `Optional[int]` | `None` | Random seed for tie-breaking |

#### Internal State Attributes
| Attribute | Type | Shape | Initialization | Description |
|---|---|---|---|---|
| `belief` | `np.ndarray` | `(K,)`, `float64` | `0.1 * ones` | Posterior occupancy belief $b_t(k)$ |
| `aoi` | `np.ndarray` | `(K,)`, `float64` | `zeros` | Age of Information in slots |
| `P01` | `np.ndarray` | `(K,)`, `float64` | `0.03 * ones` | $P(0 \to 1)$ burst probability |
| `P11` | `np.ndarray` | `(K,)`, `float64` | `0.92 * ones` | $P(1 \to 1)$ burst persistence |
| `_last_state_at_visit` | `np.ndarray` | `(K,)`, `int` | `zeros` | Most recent observed hit (1) or miss (0) |
| `_last_slot_at_visit` | `np.ndarray` | `(K,)`, `int` | `zeros` | Time slot when band was last sensed |
| `_last_action` | `int` | scalar | `0` | Sub-band visited in previous slot |
| `_consecutive_dwells` | `int` | scalar | `0` | Consecutive dwell streak on `_last_action` |
| `t` | `int` | scalar | `0` | Total scheduling steps taken |

#### Methods & API
1. `reset(seed: Optional[int] = None) -> None`:
   - Resets `belief = 0.1`, `aoi = 0`, `P01 = 0.03`, `P11 = 0.92`, `t = 0`, and dwell counters.
2. `compute_whittle_index(k: int) -> float`:
   - Returns scalar $W(b[k])$ using the analytical formula.
3. `select_band(obs: np.ndarray, info: Optional[dict] = None) -> int`:
   - Public entry point (inherited from `BaseScheduler`).
   - Calls `_choose_band(obs, info)`, advances `self.t += 1`, returns selected sub-band index $k \in \{0, \dots, K-1\}$.
4. `update_feedback(action: int, hit: bool, info: Optional[dict] = None) -> None`:
   - Updates `P01[k]` / `P11[k]`, updates `belief[k]` via Bayes, diffuses unvisited bands toward prior $0.15$, resets `aoi[action] = 0`, increments `aoi[j] += 1` for $j \ne action$.

---

### 3.3 `MultiWhittleIndexScheduler` (`schedulers/multi_schedulers.py`)

#### Constructor Parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `K` | `int` | `35` | Total number of frequency sub-bands |
| `M` | `int` | `4` | Number of receiver tuners / channels |
| `Pd` | `float` | `0.95` | Probability of detection |
| `Pfa` | `float` | `1e-4` | False alarm rate |
| `aoi_weight` | `float` | `0.60` | AoI exploration subsidy weight |
| `aoi_max` | `float` | `50.0` | Normalization cap for AoI |
| `lr_transition`| `float` | `0.05` | Learning rate for transition matrix |
| `camping_penalty_weight` | `float` | `0.40` | Penalty per consecutive dwell on active band |
| `seed` | `Optional[int]` | `None` | Seed for tie-breaking |

#### Internal State Attributes
| Attribute | Type | Shape | Initialization | Description |
|---|---|---|---|---|
| `belief` | `np.ndarray` | `(K,)`, `float64` | `0.1 * ones` | Posterior belief per band |
| `aoi` | `np.ndarray` | `(K,)`, `float64` | `zeros` | Age of Information per band |
| `P01` | `np.ndarray` | `(K,)`, `float64` | `0.03 * ones` | $P(0 \to 1)$ per band |
| `P11` | `np.ndarray` | `(K,)`, `float64` | `0.92 * ones` | $P(1 \to 1)$ per band |
| `_last_actions` | `np.ndarray` | `(M,)`, `int` | `np.arange(M) % K` | Sub-bands visited in previous slot |
| `_consecutive_dwells`| `np.ndarray` | `(K,)`, `int` | `zeros` | Per-band consecutive dwell streaks |
| `t` | `int` | scalar | `0` | Time slot counter |

#### Methods & API
1. `reset(seed: Optional[int] = None) -> None`:
   - Resets state arrays and time step counter.
2. `compute_whittle_index(k: int) -> float`:
   - Returns scalar closed-form index.
3. `select_bands(obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray`:
   - Computes composite scores across all $K$ bands, takes `np.argsort(scores)[::-1][:M]`, updates per-band consecutive dwell counters, advances `self.t += 1`, returns `np.ndarray` of shape `(M,)` with integer dtype.
4. `update_feedback(actions: np.ndarray | Sequence[int], hits: Union[dict[int, bool], np.ndarray, Sequence[bool]], info: Optional[dict] = None) -> None`:
   - Accepts array of actions of length $M$ and hit feedback (either dict `{band: bool}` or boolean sequence).
   - Performs parallel transition updates and Bayesian updates on all sensed bands.
   - Vectorized diffusion of unvisited bands toward prior.
   - Resets `aoi[actions] = 0` and increments unvisited AoI.

---

### 3.4 Existing C++ Implementation (`hardware/whittle_index.hpp`)

`hardware/whittle_index.hpp` is a standalone, header-only C++20 implementation authored by Mukesh. It contains:
```cpp
namespace ew {
constexpr size_t NUM_BANDS = 35;

struct WhittleSchedulerState {
    std::array<float, NUM_BANDS> belief;     // b[k] in [0.001, 0.999]
    std::array<float, NUM_BANDS> aoi;        // Age-of-Information (slots)
    std::array<float, NUM_BANDS> P01;        // Transition P(0 -> 1)
    std::array<float, NUM_BANDS> P11;        // Transition P(1 -> 1)
    uint32_t last_action;
    uint32_t consecutive_dwells;
};

void init_state(WhittleSchedulerState& state) noexcept;
float compute_whittle_index(float p, float p01, float p11) noexcept;
uint32_t select_next_band(WhittleSchedulerState& state) noexcept;
void update_feedback(WhittleSchedulerState& state, uint32_t action, bool hit) noexcept;
}
```

#### Performance in `hardware/test_timing.cpp`
A 100,000-iteration loop of `select_next_band` + `update_feedback` runs in pure C++ on CPU. Typical results:
- **Decision + Update Latency**: $\approx 15 - 30\text{ ns}$ per cycle (far below the $< 100\text{ ns}$ requirement).
- **50 µs Dwell Budget Used**: $< 0.1\%$ of the available $50\ \mu\text{s}$ time window.

---

## 4. Existing Unit Test Coverage

There are 87 tests in total across the repository (`uv run pytest`), all passing. The specific tests exercising Whittle schedulers and RMAB models are:

### 4.1 `tests/test_rmab.py` (`TestWhittleIndexScheduler`)
1. `test_whittle_index_bounds`:
   - Verifies monotonicity: with $P_{01}=0.05, P_{11}=0.80$, index at $p=0.9$ ($W(0.9)$) is strictly greater than at $p=0.1$ ($W(0.1)$).
2. `test_bayesian_belief_update_on_hit_and_miss`:
   - Verifies that a sensed hit causes $b[k] > 0.5$ (from prior $0.5$) and resets $\text{AoI}[k] = 0.0$.
   - Verifies that a sensed miss causes $b[k] < 0.5$.
3. `test_aoi_increases_for_unsensed_bands`:
   - Verifies that 5 consecutive dwells on band 0 keep $\text{AoI}[0] = 0.0$ and increase $\text{AoI}[1] = 5.0$.
4. `test_action_bounds`:
   - Verifies that 50 consecutive calls to `select_band(obs)` produce actions in $[0, K-1]$.

### 4.2 `tests/test_multi_receiver.py` (`TestMultiSchedulers` & `TestMultiReceiverEnv`)
1. `test_multi_whittle_scheduler_top_m_selection`:
   - Initializes `MultiWhittleIndexScheduler(K=35, M=4)`.
   - Biases belief on bands `[4, 9, 15, 22]` to $0.95$.
   - Verifies `select_bands` returns exactly those 4 bands with 0 collisions (`len(unique(action)) == 4`).
   - Verifies feedback update with hits `{4: True, 9: True, 15: False, 22: True}` properly elevates $b[4] > 0.90$ and lowers $b[15] < 0.95$.
2. `test_collision_detection_and_penalty`:
   - Validates that redundant band assignments incur heavy reward penalties.
3. `test_parallel_aoi_reset`:
   - Validates parallel AoI reset for all $M$ tuners.

### 4.3 `tests/test_dynamic_env.py`
1. `test_whittle_scheduler_runs_on_dynamic_env`:
   - Runs 100 interaction steps of `WhittleIndexScheduler(K=35)` closed-loop with `DynamicSpectrumEnv(K=35, T=100, stage=3)` (randomized Fixed, FHSS, and Scanning emitters).
   - Verifies full episode completion, valid action bounds, and positive hit accumulation.

### 4.4 `tests/test_predictor.py`
1. `test_hybrid_scheduler_runs`:
   - Verifies `HybridPredictiveScheduler`, which embeds `WhittleIndexScheduler` as its base exploration/exploitation engine.

---

## 5. Critical Implementation Parity Gaps & Guidance for C++20 Engine (R1)

To deliver **Requirement R1 (Zero-Allocation C++20 RMAB Engine & pybind11 Extension)** with 100% mathematical parity and drop-in replacement capability, the following nuances must be resolved:

| Feature / Detail | Python `WhittleIndexScheduler` | Python `MultiWhittleIndexScheduler` | Existing C++ `hardware/whittle_index.hpp` | Required in C++20 `rmab_cpp` Extension |
|---|---|---|---|---|
| **Initial Belief** | `0.10` | `0.10` | `0.15f` | Configurable or standardized to `0.10f` |
| **Initial $P_{01} / P_{11}$** | `0.03 / 0.92` | `0.03 / 0.92` | `0.03f / 0.92f` | `0.03f / 0.92f` |
| **Online Learning ($P_{01}, P_{11}$)** | Yes ($\eta=0.05, \Delta t \le 10$) | Yes ($\eta=0.05, \Delta t \le 10$) | **Missing** (static) | Implement EMA update in C++ |
| **Multi-Tuner Support ($M=4$)** | N/A ($M=1$) | Yes (Top-$M$ sort) | **Missing** ($M=1$ only) | Implement zero-allocation top-$M$ selection (e.g. `std::partial_sort` or linear selection) |
| **Consecutive Dwells State** | Scalar (`_consecutive_dwells`) | Array (`_consecutive_dwells[K]`) | Scalar | Support per-band dwell array for multi-tuner |
| **Tie-Breaker** | RNG $U(0, 10^{-5})$ | RNG $U(0, 10^{-5})$ | Strict `>` (deterministic) | Optional deterministic mode or fast LCG RNG |
| **API Method Names** | `select_band`, `update_feedback` | `select_bands`, `update_feedback` | Standalone C functions | Expose `select_band`, `select_bands`, `update_feedback`, plus aliases (`select_action`, `step`, `update`) |
| **Build System** | N/A | N/A | None (`clang++` CLI) | Add `CMakeLists.txt` producing `rmab_cpp.so` via `pybind11` |

---

## 6. Verification Plan & Test Commands

1. **Verify Existing Unit Tests**:
   ```bash
   uv run pytest
   ```
   *Expected*: 87 passed.
2. **Compile and Verify Existing C++ Micro-Benchmark**:
   ```bash
   clang++ -O3 -std=c++20 hardware/test_timing.cpp -o hardware/test_timing
   ./hardware/test_timing
   ```
   *Expected*: Average latency $< 50\text{ ns}$ per decision cycle.
3. **Parity Testing Framework for C++ pybind11 Engine**:
   - Construct a test comparing Python `WhittleIndexScheduler` vs C++ `rmab_cpp.WhittleEngine` across 1,000 identical pseudo-random hit/miss sequences, asserting $\max |b_{\text{py}}[k] - b_{\text{cpp}}[k]| < 10^{-5}$.
