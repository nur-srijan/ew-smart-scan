# DRDO EW Smart Scan — Demo Video Script & Transcript
**Problem Statement ID:** 26055 | **Project:** Autonomous Spectrum Surveillance & Multi-Receiver Scheduling  
**Target Audience:** DRDO Evaluators, Hackathon Jury, Defense SIGINT Technical Committee  
**Total Duration:** ~3 Minutes (180 Seconds)  

---

## 🎬 Video Production Overview

| Segment | Timing | Key Visual / Screen Focus | Core Message |
|---|---|---|---|
| **1. Hook & Problem** | `0:00 - 0:30` | Dashboard Header & Title Strip | Traditional EW blind spots in 0.5–18 GHz agile spectrum |
| **2. Fleet Architecture** | `0:30 - 0:55` | **Tab 02: Fleet Telemetry** | 3 distributed nodes (2 UAVs + 1 Ground Station), 4 tuners, 0.0% collisions |
| **3. Live Spectrum & Waterfall** | `0:55 - 1:30` | **Tab 01: Live Waterfall** | Real-time 35-band canvas, multi-tuner bracketing, 7×5 AoI grid |
| **4. EOB Threat Tracking & PDW** | `1:30 - 2:05` | **Tab 03: Threat Library** | Online PRI extraction (S-300, Krasukha-4), one-click PDW CSV/JSON export |
| **5. Performance Metrics & FoM** | `2:05 - 2:35` | **Tab 04: Performance & FoM** | 4 KPI cards (22.2% IR, 0.237s TTI), 4 interactive Plotly charts, Monte Carlo table |
| **6. Mission Control & Edge C++20** | `2:35 - 2:50` | **Tab 05: Mission Control** + CLI | Real-time policy switching, 20.8 ns C++20 engine, 50µs timing deadline |
| **7. Conclusion & Operational Impact** | `2:50 - 3:00` | Full TOC Tactical Overview | Zero-prior autonomous SIGINT ready for NVIDIA Jetson Orin Nano |

---

## 📜 Full Voiceover Transcript with Visual Directives

### [0:00 - 0:30] Segment 1: The Tactical Dilemma & Introduction

**Screen Action:**
- Open browser at `http://127.0.0.1:8050`.
- Show the dark tactical header: **C2-ESM: Autonomous Spectrum Surveillance & Telemetry Center**.
- Mouse hovers over the top status strip: `35 sub-bands`, `0.5–18 GHz`, `0.0% Tuner Collisions`, `SYSTEM ARMED`.

**Voiceover (Speaker):**
> "Modern electronic warfare presents a severe operational challenge: an instantaneous surveillance span across 0.5 to 18 Gigahertz, contested by frequency-agile radars, dense jammers, and rotating scanning emitters.
>
> Traditional receiver sweeping policies — like sequential or pseudo-random sweeps — suffer from blind spots, high intercept latencies, and an interception ratio often below three percent.
>
> Today, we present **Smart Scan EW**: an autonomous, multi-payload cooperative scheduling and telemetry platform designed for DRDO Problem Statement 1778. Operating with **zero prior emitter intelligence**, it dynamically coordinates multiple electronic support tuners using Restless Multi-Armed Bandits and cooperative role scheduling."

---

### [0:30 - 0:55] Segment 2: Fleet Architecture & Multi-Payload Allocation

**Screen Action:**
- Click on **Tab 02: Fleet Telemetry**.
- Pan across the three node telemetry cards:
  - **Node Alpha (UAV-1)**: Tuner 0 (Phase-Locked Pulse Tracker).
  - **Node Bravo (UAV-2)**: Tuners 1 & 2 (Agile FHSS Chaser Pair).
  - **Node Charlie (Ground Station)**: Tuner 3 (Wideband AoI Sentry).
- Show the live battery gauges, RF synthesizer temperature, and the lower **Receiver & Scheduler State Table**.
- Click the **Acknowledge advisory** button on Node Bravo to show live operator responsiveness.

**Voiceover (Speaker):**
> "In a contested theater, surveillance isn't conducted by an isolated receiver. Our solution orchestrates a distributed multi-receiver mesh across three autonomous nodes:
>
> - **Node Alpha**, an airborne standoff UAV, runs **Tuner 0**, dedicated to phase-locked delta-TOA pulse tracking on periodic radars.
> - **Node Bravo**, a penetrating recon UAV, deploys **Tuners 1 and 2** as an agile chaser pair, tracking online Markov transition hops of hostile frequency-hopping jammers like the Krasukha-4.
> - And **Node Charlie**, our ground station, operates **Tuner 3** as a wideband sentry, minimizing spectrum Age-of-Information.
>
> Crucially, our cooperative scheduler guarantees **0.0% tuner collisions** at all times through strictly disjoint frequency assignments."

---

### [0:55 - 1:30] Segment 3: Live Spectrum Waterfall & AoI Grid

**Screen Action:**
- Switch to **Tab 01: Live Waterfall**.
- Switch the **Receiver View** dropdown from *Fleet Composite* to *Node Alpha*, then back to *Fleet Composite*.
- Point out the 35 sub-band waterfall canvas:
  - Cyan blocks (Tuner 0 hits)
  - Amber & Green blocks (Tuners 1 & 2 bracketing FHSS hops)
  - Purple blocks (Tuner 3 sentry sweeps)
- Hover over the **7×5 Sub-Band Observation Age (AoI) Grid**, highlighting outlined active bands and step counters.
- Point to the **Current Receiver Observation** card showing Bayesian belief probability and detector status.

**Voiceover (Speaker):**
> "Here on the **Live Waterfall** display, we observe the tactical spectrum in real-time across all 35 channels. 
>
> Our canvas renders 50-microsecond dwell windows with color-coded tuner overlays. 
>
> Watch as Tuners 1 and 2 dynamically bracket the hop tracks of agile emitters across bands 7 through 26, while Tuner 0 maintains a solid lock on high-priority fixed emitters.
>
> Beneath the waterfall, the **Age-of-Information matrix** monitors the freshness of every 500-Megahertz sub-band. If an emitter ceases transmission, the scheduler instantly shifts from exploitation to exploration, ensuring no sector of the battlefield remains unobserved."

---

### [1:30 - 2:05] Segment 4: Electronic Order of Battle (EOB) & PDW SIGINT Export

**Screen Action:**
- Switch to **Tab 03: Threat Library**.
- Highlight the **Electronic Order of Battle (EOB)** table:
  - Active threats updating live: *RADAR-01 [S-300 PMU-2]*, *RADAR-03 [Krasukha-4]*, *RADAR-04 [Su-35S Irbis-E]*.
  - Point to the **Estimated PRI** column (e.g., *5,250.0 µs*, *2,100.0 µs*).
  - Point to the **Alert Level** badges (*CRITICAL*, *HIGH*).
- Scroll down to the **Intercepted Pulse Descriptor Words (PDW Stream)**.
- Click **📥 Export PDW (CSV)** — show the downloaded file notification.
- Click **📥 Export PDW (JSON)**.

**Voiceover (Speaker):**
> "Without requiring any pre-loaded threat libraries, Smart Scan constructs a live **Electronic Order of Battle** through online signal analysis.
>
> By computing median pulse intervals from successive Time-of-Arrival hits, the engine estimates the pulse repetition interval in microseconds, identifies agility hop sets, and flags threat levels in real time — correctly categorizing air defense radars, airborne fire control systems, and tactical jammers.
>
> For downstream SIGINT exploitation, every intercepted pulse is parsed into standard **Pulse Descriptor Words** — capturing exact timestamp, receiver node, carrier frequency, pulse width, and RSSI — exportable with a single click in CSV and JSON formats."

---

### [2:05 - 2:35] Segment 5: Performance Metrics, FoM Benchmarks & Figures of Merit

**Screen Action:**
- Switch to **Tab 04: Performance & FoM**.
- Point to the **Top 4 KPI Metric Cards**:
  - `Global IR: 22.17%` (`+737% vs M=1 Baseline`)
  - `Mean TTI: 0.237 s` (`30% Threat Latency Reduction`)
  - `Throughput: 175.3 pulses` (`5.9x Multiplier`)
  - `Collision Rate: 0.0%` (`Guaranteed Disjoint Allocations`)
- Pan across the **Dual Plotly Charts (Row 1)**:
  - Hover over the **Cumulative Pulse Interceptions** curve showing Cooperative AI pulling sharply ahead of Multi-Whittle, Multi-Sequential, and PseudoRandom.
  - Hover over the **Policy IR % Comparison** bar chart (from 2.65% up to 22.17%).
- Pan across the **Dual Plotly Charts (Row 2)**:
  - Point out the **35 Sub-Band Dwell Distribution Histogram** (demonstrating focused surveillance on threat bands with non-zero AoI exploration baseline).
  - Point to the **Time-to-Intercept (TTI)** horizontal bar chart.
- Show the **Rigorous Monte Carlo Benchmark Evaluation Matrix Table** (20 Dynamic Episodes).

**Voiceover (Speaker):**
> "Under **Performance & Figures of Merit**, we evaluate our system against rigorous empirical benchmarks.
>
> In 20-episode Monte Carlo evaluations across dynamic multi-emitter threats, our cooperative policy achieves a **22.17% global interception ratio** — an **8x improvement** over single-receiver sweeps and nearly **6x higher** than sequential baselines, while slashing Time-to-Intercept by 30 percent.
>
> As shown in the cumulative discovery curve, the cooperative policy detects pulses earlier and maintains steady acquisition throughput without plateauing.
>
> The sub-band dwell histogram demonstrates our anti-camping guarantee: while tuners prioritize active threats, AoI exploration subsidies ensure continuous coverage across all 35 channels.
>
> Every metric shown is generated through standard gym-compliant simulation and verified across multi-episode Monte Carlo runs."

---

### [2:35 - 2:50] Segment 6: Mission Control & Edge C++20 Execution

**Screen Action:**
- Switch to **Tab 05: Mission Control**.
- Show the **Scheduler Algorithm** dropdown: switch from `Cooperative Role Scheduler` to `Multi-Receiver Whittle RMAB` or `Multi-Sequential Sweep`.
- Show the **Scenario Preset** dropdown: show `Contested Dense Agile (4 Emitters)` vs `Electronic Attack Strobe`.
- Click **Apply Config** or toggle the speed slider (1x / 2x / 5x / 10x).
- Point to the live **System Event Log** recording policy shifts.
- (Optional brief cut) Show terminal running `./hardware/test_timing`:
  ```bash
  ./hardware/test_timing
  # Output: 20.80 ns Decision Latency | 0 Deadline Misses
  ```

**Voiceover (Speaker):**
> "In **Mission Control**, operators can dynamically reconfigure scheduling policies, inject contested jamming scenarios, or throttle simulation speeds on the fly.
>
> Crucially, for edge deployment on platforms like the NVIDIA Jetson Orin Nano, our zero-allocation C++20 scheduling core executes ranking decisions in just **20.8 nanoseconds** — running well within the strict 50-microsecond dwell window with **zero deadline misses**."

---

### [2:50 - 3:00] Segment 7: Summary & Tactical Impact

**Screen Action:**
- Switch back to **Tab 01: Live Waterfall** or full TOC view.
- Let the canvas run smoothly with live hits and telemetry counters updating.
- Display closing banner:
  - *Smart Scan EW — Team Robotics & Drones*
  - *SIH 2026 | DRDO Problem Statement 1778*
  - *Production Ready: 279/279 Passing Tests | C++20 + Python + Pybind11*

**Voiceover (Speaker):**
> "Smart Scan EW bridges theoretical multi-armed bandit mathematics with real-world radar electronic support requirements:
>
> - Zero prior intelligence.
> - Multi-platform cooperative fleet coordination.
> - Sub-microsecond edge-native execution.
> - And military-grade tactical command visualization.
>
> Thank you."

---

## 💡 Practical Recording Tips for the Presenter

1. **Browser Resolution & Scaling:**
   - Fullscreen browser at 1080p (1920×1080) or 2K.
   - Zoom level: `90%` or `100%` so all cards and tables fit cleanly without awkward scrolling.
2. **Terminal Quick-Check (for C++ B-roll):**
   ```bash
   cd /Users/nursrijan/dev/sih-project
   ./hardware/test_timing
   ```
   *Displays the 20.8 ns decision latency and 0 deadline misses cleanly.*
3. **Pacing & Cursor Cues:**
   - Keep cursor movements smooth and deliberate — hover over the exact metric cards or chart series being mentioned.
   - Tab switching sequence: **Tab 02** $\rightarrow$ **Tab 01** $\rightarrow$ **Tab 03** $\rightarrow$ **Tab 04** $\rightarrow$ **Tab 05** $\rightarrow$ **Tab 01**.
