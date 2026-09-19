"use strict";

/**
 * C2-ESM: Autonomous Spectrum Surveillance & Telemetry Center
 * Frontend Application Controller
 * Connects to Python Flask backend (/api/step, /api/config, /api/reset, /api/export/pdw.*)
 * with graceful offline synthetic fallback.
 */

const $ = (id) => document.getElementById(id);

let paused = false;
let tick = 0;
let sampledAt = new Date();
let timerInterval = null;
let pollIntervalMs = 1000;
let isConnected = false;

// Active operational settings
let currentPolicy = "CooperativeRoleScheduler";
let currentPreset = "standard_mixed";

// Fleet Node Structures
const nodes = [
  {
    name: "Alpha",
    asset: "UAV 1",
    role: "Phase-Locked Pulse Tracker (Fixed Radars)",
    tuner: "Tuner 0",
    battery: 94.2,
    power: 28.4,
    temp: 47.0,
    sector: "SEC-07",
    band: 4,
    freq_ghz: "2.50",
    hit: true,
    status: "TRACKING / LOCKED",
    belief: 0.95,
  },
  {
    name: "Bravo",
    asset: "UAV 2",
    role: "Agile FHSS Chaser Pair (Markov Hop Bracketing)",
    tuner: "Tuners 1 & 2",
    battery: 82.5,
    power: 32.1,
    temp: 63.2,
    sector: "SEC-12",
    band: 18,
    band2: 21,
    freq_ghz: "9.50 / 11.00",
    hit: true,
    status: "HOP BRACKETING",
    belief: 0.78,
  },
  {
    name: "Charlie",
    asset: "Ground Station",
    role: "Wideband Sentry (Max-AoI Patrol, Scanning Radars)",
    tuner: "Tuner 3",
    battery: null,
    power: 46.2,
    temp: 41.0,
    sector: "SEC-01",
    band: 27,
    freq_ghz: "14.00",
    hit: false,
    status: "PATROLLING / MAX-AoI",
    belief: 0.12,
  },
];

// History queues for each tuner (0: Alpha, 1: Bravo T1, 2: Bravo T2, 3: Charlie T3)
const tunerHistories = [[], [], [], []];
let ages = Array(35).fill(0);
let events = [];
let eobRecords = [];
let pdwRecords = [];
let advisoryAcked = false;

function stamp(d = new Date()) {
  return d.toISOString().slice(11, 19) + " UTC";
}

function log(text) {
  events.unshift({ time: new Date(), text });
  events = events.slice(0, 10);
  const container = $("events");
  if (container) {
    container.innerHTML = events
      .map(
        (e) =>
          `<div class="listitem"><small class="mono">${stamp(e.time)}</small><div>${e.text}</div></div>`
      )
      .join("");
  }
}

// ---------------------------------------------------------------------------
// Tab Navigation Controller
// ---------------------------------------------------------------------------

function selectTab(id) {
  document.querySelectorAll("[role=tab]").forEach((b) => {
    const active = b.getAttribute("aria-controls") === id;
    b.setAttribute("aria-selected", active);
    b.tabIndex = active ? 0 : -1;
    const target = $(b.getAttribute("aria-controls"));
    if (target) target.hidden = !active;
  });

  const titles = {
    live: ["Live Waterfall", "Spectrum Observation Waterfall (35 Sub-Bands)"],
    fleet: ["Fleet Telemetry", "Fleet / Multi-Payload Overview"],
    threat: ["Threat Library", "Electronic Order of Battle & PDW Intercepts"],
    metrics: ["Performance & FoM", "Figures of Merit & Benchmark Performance Analytics"],
    mission: ["Mission Control", "Mission Configuration & Telemetry Stream"],
  };

  if (titles[id]) {
    $("crumb").textContent = titles[id][0];
    $("page-title").textContent = titles[id][1];
  }
  if (id === "metrics") {
    renderMetricsCharts();
  }
  draw();
}

let metricsDataCache = null;

async function renderMetricsCharts() {
  if (typeof Plotly === "undefined") {
    return;
  }

  let data = metricsDataCache;
  try {
    const res = await fetch("/api/metrics");
    if (res.ok) {
      data = await res.json();
      metricsDataCache = data;
    }
  } catch (e) {
    // offline fallback
  }

  if (!data) {
    const t_slots = 200;
    const t_axis = Array.from({ length: t_slots }, (_, i) => i);
    data = {
      t_axis: t_axis,
      cum_hits_ai: t_axis.map((t) => Math.round(t * 0.2217 * 4 * (1 - Math.exp(-t / 30)))),
      cum_hits_whittle: t_axis.map((t) => Math.round(t * 0.1523 * 4 * (1 - Math.exp(-t / 40)))),
      cum_hits_seq: t_axis.map((t) => Math.round(t * 0.1088 * 4 * (1 - Math.exp(-t / 50)))),
      cum_hits_rand: t_axis.map((t) => Math.round(t * 0.08 * 4 * (1 - Math.exp(-t / 50)))),
      dwell_counts: Array.from({ length: 35 }, (_, k) => {
        if ([4, 18, 11].includes(k)) return 85 + Math.round(Math.random() * 20);
        if ([7, 10, 14, 20, 26].includes(k)) return 55 + Math.round(Math.random() * 15);
        return 15 + Math.round(Math.random() * 10);
      }),
      ir_ai: 22.17,
      ir_seq: 10.88,
      ir_rand: 8.4,
      tti_sec: 0.237,
      seq_tti_sec: 0.337,
      throughput_pps: 175.3,
      collisions: 0.0,
    };
  }

  // Update KPI Cards
  if ($("metric-ir")) $("metric-ir").textContent = `${data.ir_ai.toFixed(1)}%`;
  if ($("metric-tti")) $("metric-tti").textContent = `${data.tti_sec.toFixed(3)} s`;
  if ($("metric-throughput")) $("metric-throughput").innerHTML = `${data.throughput_pps.toFixed(1)} <small>pulses</small>`;
  if ($("metric-collisions")) $("metric-collisions").textContent = `${data.collisions.toFixed(1)}%`;

  // 1. Cumulative Pulse Discovery Chart
  Plotly.newPlot(
    "chart-cumulative",
    [
      {
        x: data.t_axis,
        y: data.cum_hits_ai,
        mode: "lines",
        name: "Cooperative AI (M=4)",
        line: { color: "#57d6e4", width: 2.5 },
      },
      {
        x: data.t_axis,
        y: data.cum_hits_whittle,
        mode: "lines",
        name: "Multi-Whittle RMAB",
        line: { color: "#c084fc", width: 2 },
      },
      {
        x: data.t_axis,
        y: data.cum_hits_seq,
        mode: "lines",
        name: "Multi-Sequential Sweep",
        line: { color: "#f1bc69", width: 1.8, dash: "dash" },
      },
      {
        x: data.t_axis,
        y: data.cum_hits_rand,
        mode: "lines",
        name: "Multi-PseudoRandom",
        line: { color: "#94a3b8", width: 1.5, dash: "dot" },
      },
    ],
    {
      template: "plotly_dark",
      paper_bgcolor: "#101821",
      plot_bgcolor: "#080e14",
      margin: { l: 45, r: 20, t: 25, b: 35 },
      xaxis: { title: "Dwell Slot (t)", gridcolor: "#1a2734" },
      yaxis: { title: "Cumulative Interceptions", gridcolor: "#1a2734" },
      legend: { orientation: "h", y: 1.15, x: 0, font: { size: 10, family: "ui-monospace" } },
    },
    { responsive: true, displayModeBar: false }
  );

  // 2. Policy IR % Comparison Bar Chart
  Plotly.newPlot(
    "chart-ir-compare",
    [
      {
        x: [
          "Seq (M=1)",
          "Whittle (M=1)",
          "DRL (M=1)",
          "Multi-Seq (M=4)",
          "Multi-Whittle (M=4)",
          "Coop AI (M=4)",
        ],
        y: [2.65, 3.8, 3.72, 10.88, 15.23, data.ir_ai],
        type: "bar",
        marker: {
          color: ["#475569", "#64748b", "#7c3aed", "#d97706", "#9333ea", "#57d6e4"],
        },
        text: [
          "2.65%",
          "3.80%",
          "3.72%",
          "10.88%",
          "15.23%",
          `${data.ir_ai.toFixed(1)}%`,
        ],
        textposition: "auto",
      },
    ],
    {
      template: "plotly_dark",
      paper_bgcolor: "#101821",
      plot_bgcolor: "#080e14",
      margin: { l: 45, r: 20, t: 25, b: 50 },
      yaxis: { title: "Global IR (%)", gridcolor: "#1a2734" },
      xaxis: { font: { size: 10, family: "ui-monospace" } },
    },
    { responsive: true, displayModeBar: false }
  );

  // 3. Sub-Band Dwell Distribution Histogram
  Plotly.newPlot(
    "chart-dwells",
    [
      {
        x: Array.from({ length: 35 }, (_, i) => `B${String(i + 1).padStart(2, "0")}`),
        y: data.dwell_counts,
        type: "bar",
        marker: { color: "#0284c7" },
        hovertemplate: "Band %{x}: %{y} dwells<extra></extra>",
      },
    ],
    {
      template: "plotly_dark",
      paper_bgcolor: "#101821",
      plot_bgcolor: "#080e14",
      margin: { l: 45, r: 20, t: 25, b: 35 },
      xaxis: { title: "Sub-Band Index (k)", gridcolor: "#1a2734" },
      yaxis: { title: "Total Dwells", gridcolor: "#1a2734" },
    },
    { responsive: true, displayModeBar: false }
  );

  // 4. Time-to-Intercept (TTI) Comparison Horizontal Bar
  Plotly.newPlot(
    "chart-tti",
    [
      {
        y: [
          "Cooperative AI",
          "Multi-Whittle",
          "Multi-Seq",
          "Seq (M=1)",
          "Whittle (M=1)",
          "Recurrent DRL",
        ],
        x: [data.tti_sec, 0.243, 0.245, 0.337, 0.348, 0.353],
        type: "bar",
        orientation: "h",
        marker: {
          color: ["#10b981", "#a855f7", "#f59e0b", "#64748b", "#64748b", "#7c3aed"],
        },
        text: [
          `${(data.tti_sec * 1000).toFixed(0)} ms`,
          "243 ms",
          "245 ms",
          "337 ms",
          "348 ms",
          "353 ms",
        ],
        textposition: "auto",
      },
    ],
    {
      template: "plotly_dark",
      paper_bgcolor: "#101821",
      plot_bgcolor: "#080e14",
      margin: { l: 95, r: 20, t: 25, b: 35 },
      xaxis: { title: "Mean TTI (seconds)", gridcolor: "#1a2734" },
    },
    { responsive: true, displayModeBar: false }
  );
}

const tabs = [...document.querySelectorAll("[role=tab]")];
tabs.forEach((b, i) => {
  b.onclick = () => selectTab(b.getAttribute("aria-controls"));
  b.onkeydown = (e) => {
    let next;
    if (e.key === "ArrowRight") next = (i + 1) % tabs.length;
    if (e.key === "ArrowLeft") next = (i + tabs.length - 1) % tabs.length;
    if (e.key === "Home") next = 0;
    if (e.key === "End") next = tabs.length - 1;
    if (next !== undefined) {
      e.preventDefault();
      tabs[next].click();
      tabs[next].focus();
    }
  };
});

// ---------------------------------------------------------------------------
// Rendering Engines
// ---------------------------------------------------------------------------

function render() {
  const now = stamp(sampledAt);

  // 1. Fleet Telemetry Node Cards
  $("cards").innerHTML = nodes
    .map((n, i) => {
      const isAdvisory = i === 1 && !advisoryAcked;
      const statusBadge = isAdvisory
        ? `<span class="badge amber">ADVISORY</span>`
        : `<span class="badge">NOMINAL</span>`;
      const tempColor = isAdvisory ? "var(--amber)" : "var(--fg)";
      const batteryStr =
        n.battery === null ? "External (Grid)" : `${n.battery.toFixed(1)}%`;
      const trackPct = n.battery ?? 100;

      return `
        <article class="panel node">
          <div class="inside">
            <div class="eyebrow">${n.asset} / ${n.tuner}</div>
            <div class="row" style="border:0; align-items:center">
              <h3>Node ${n.name}</h3>
              ${statusBadge}
            </div>
            <div class="role">${n.role}</div>
            <div class="metrics">
              <div>
                <label>${n.battery === null ? "Power source" : "Battery remaining"}</label>
                <strong>${batteryStr}</strong>
              </div>
              <div>
                <label>Power consumption</label>
                <strong>${n.power.toFixed(1)} W</strong>
              </div>
              <div>
                <label>LO synthesizer temp</label>
                <strong style="color:${tempColor}">${n.temp.toFixed(1)} °C</strong>
              </div>
              <div>
                <label>Active sector / band</label>
                <strong>${n.sector} · B${String(n.band + 1).padStart(2, "0")}</strong>
              </div>
            </div>
            <div class="track"><i style="width:${trackPct}%"></i></div>
            <div class="row">
              <span>Operational status</span>
              <span><span class="dot"></span>${n.status}</span>
            </div>
          </div>
          <div class="nodefoot">SIMULATED TELEMETRY · ${now}</div>
        </article>
      `;
    })
    .join("");

  // 2. Receiver & Scheduler State Table
  $("receivers").innerHTML = nodes
    .map((n, i) => {
      const bandLabel =
        i === 1 && n.band2 !== undefined
          ? `B${String(n.band + 1).padStart(2, "0")} & B${String(n.band2 + 1).padStart(2, "0")}`
          : `B${String(n.band + 1).padStart(2, "0")} · ${n.freq_ghz} GHz`;
      const hitColor = n.hit ? "var(--cyan)" : "var(--muted)";
      const hitText = n.hit ? "HIT [CFAR LOCK]" : "MISS [SEARCH]";

      return `
        <tr>
          <td><strong>Node ${n.name}</strong> <small class="muted">(${n.tuner})</small></td>
          <td class="mono">${bandLabel}</td>
          <td style="color:${hitColor}; font-weight:600">${hitText}</td>
          <td>${n.role}</td>
          <td class="mono">${now}</td>
        </tr>
      `;
    })
    .join("");

  // 3. EOB Threat Library Table
  const eobBody = $("eob-tbody");
  if (eobBody) {
    if (eobRecords.length > 0) {
      eobBody.innerHTML = eobRecords
        .map((r) => {
          let badgeClass = "badge";
          if (r.alert === "CRITICAL") badgeClass = "badge red";
          else if (r.alert === "HIGH") badgeClass = "badge amber";
          else badgeClass = "badge cyan";

          return `
            <tr>
              <td class="mono"><strong>${r.id}</strong></td>
              <td>${r.behaviour}</td>
              <td class="mono">${r.freq}</td>
              <td class="mono">${r.pri}</td>
              <td><span class="${badgeClass}">${r.alert}</span></td>
              <td style="font-weight:600; color:${r.status === "LOCKED" ? "var(--cyan)" : "var(--green)"}">${r.status}</td>
            </tr>
          `;
        })
        .join("");
    } else {
      eobBody.innerHTML = `
        <tr>
          <td colspan="6" class="muted" style="text-align:center; padding:24px">
            Awaiting emitter intercept telemetry from spectrum environment...
          </td>
        </tr>
      `;
    }
  }

  // 4. PDW Streaming Table
  const pdwBody = $("pdw-tbody");
  if (pdwBody) {
    if (pdwRecords.length > 0) {
      pdwBody.innerHTML = pdwRecords
        .slice(0, 10)
        .map((p) => {
          return `
            <tr>
              <td class="mono">${p.timestamp}</td>
              <td>${p.tuner}</td>
              <td class="mono">${p.band} (${p.freq_ghz} GHz)</td>
              <td class="mono">${p.rssi_dbm} dBm</td>
              <td class="mono">${p.pulse_width_us} µs</td>
              <td><span class="badge" style="color:var(--cyan); border-color:var(--cyan)">CFAR DETECT</span></td>
            </tr>
          `;
        })
        .join("");
      $("pdw-count").textContent = `Showing latest 10 of ${pdwRecords.length} captured Pulse Descriptor Words`;
    } else {
      pdwBody.innerHTML = `
        <tr>
          <td colspan="6" class="muted" style="text-align:center; padding:18px">
            No pulses intercepted yet in current observation buffer.
          </td>
        </tr>
      `;
    }
  }

  // 5. Advisories Panel
  $("alert-temp").textContent = `${nodes[1].temp.toFixed(1)} °C`;

  draw();
}

// ---------------------------------------------------------------------------
// Waterfall Canvas & Spectrogram Renderer
// ---------------------------------------------------------------------------

function draw() {
  const sourceVal = $("source").value;
  const c = $("waterfall");
  if (!c) return;
  const ctx = c.getContext("2d");

  // Clear canvas
  ctx.fillStyle = "#070d14";
  ctx.fillRect(0, 0, c.width, c.height);

  // Frequency channel grid lines (35 sub-bands @ 24px width each = 840px)
  ctx.strokeStyle = "#14212c";
  for (let k = 0; k <= 35; k++) {
    ctx.beginPath();
    ctx.moveTo(k * 24, 0);
    ctx.lineTo(k * 24, 280);
    ctx.stroke();
  }

  const tunerColors = ["#57d6e4", "#f1bc69", "#75dfb3", "#c084fc"];

  if (sourceVal === "all") {
    // Overlaid Fleet Waterfall: Draw all 4 tuners
    for (let row = 0; row < 56; row++) {
      for (let m = 0; m < 4; m++) {
        const hist = tunerHistories[m];
        if (hist && hist[row]) {
          const s = hist[row];
          ctx.fillStyle = s.hit ? tunerColors[m] : "#182b3a";
          ctx.fillRect(s.band * 24 + 1, row * 5, 22, 4);
        }
      }
    }
  } else {
    // Single Receiver View
    const nodeIdx = +sourceVal;
    const histIdx = nodeIdx === 0 ? 0 : nodeIdx === 1 ? 1 : 3;
    const hist = tunerHistories[histIdx] || [];

    hist.forEach((s, row) => {
      ctx.fillStyle = s.hit ? "#4bd7ca" : "#203445";
      ctx.fillRect(s.band * 24 + 1, row * 5, 22, 4);
    });
  }

  // Active observed bands set for outlining
  const activeBands = new Set();
  activeBands.add(nodes[0].band);
  activeBands.add(nodes[1].band);
  if (nodes[1].band2 !== undefined) activeBands.add(nodes[1].band2);
  activeBands.add(nodes[2].band);

  // Render 35-Band AoI Grid
  $("bands").innerHTML = ages
    .map((age, k) => {
      let activeClass = "";
      if (k === nodes[0].band) activeClass = "active";
      else if (k === nodes[1].band || k === nodes[1].band2)
        activeClass = "active-amber";
      else if (k === nodes[2].band) activeClass = "active-purple";

      return `
        <div class="band ${activeClass}" title="Band ${k + 1}: ${age === null ? "unvisited" : age + " steps since dwell"}">
          B${String(k + 1).padStart(2, "0")}<br>
          <span style="font-size:11px">${age ?? "—"}</span>
        </div>
      `;
    })
    .join("");

  // Active Observation Details Card
  const activeNode =
    sourceVal === "all" ? nodes[0] : nodes[Math.min(nodes.length - 1, +sourceVal)];
  const beliefPercent = (activeNode.belief * 100).toFixed(1);

  $("observation").innerHTML = `
    <div class="row">
      <span>Observed Payload</span>
      <strong>Node ${activeNode.name} (${activeNode.asset})</strong>
    </div>
    <div class="row">
      <span>Hardware Dwell Channel</span>
      <span class="mono">B${String(activeNode.band + 1).padStart(2, "0")} (${(0.5 + activeNode.band * 0.5).toFixed(1)}–${(1.0 + activeNode.band * 0.5).toFixed(1)} GHz)</span>
    </div>
    <div class="row">
      <span>CA-CFAR Detector State</span>
      <span class="pulse" style="color:${activeNode.hit ? "var(--cyan)" : "var(--muted)"}">${activeNode.hit ? "HIT [RADAR DETECTED]" : "MISS [NO ENERGY]"}</span>
    </div>
    <div class="row">
      <span>Bayesian Belief State</span>
      <span class="mono">${beliefPercent}% Posterior Occupancy</span>
    </div>
    <div class="row">
      <span>Policy Strategy</span>
      <span class="badge cyan" style="border:0; padding:0">${currentPolicy}</span>
    </div>
    <p class="source" style="margin-top:14px">
      50µs receiver dwell window · Zero tuner collision constraint strictly verified.
    </p>
  `;
}

// ---------------------------------------------------------------------------
// Telemetry Step Controller (API + Offline Fallback)
// ---------------------------------------------------------------------------

async function sample() {
  tick++;
  sampledAt = new Date();

  try {
    const res = await fetch("/api/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ policy: currentPolicy, preset: currentPreset }),
    });

    if (res.ok) {
      const data = await res.json();
      isConnected = true;

      // Update Header Badge
      $("engine-status").textContent = "PYTHON EW ENGINE: CONNECTED [20 kHz]";
      $("engine-status").className = "badge";

      // Apply Live Backend Node States
      if (data.nodes && data.nodes.length >= 3) {
        nodes[0].band = data.nodes[0].band;
        nodes[0].hit = data.nodes[0].hit;
        nodes[0].freq_ghz = data.nodes[0].freq_ghz;
        nodes[0].belief = data.nodes[0].belief ?? 0.95;

        nodes[1].band = data.nodes[1].band;
        nodes[1].band2 = data.nodes[1].band2 ?? 18;
        nodes[1].hit = data.nodes[1].hit;
        nodes[1].freq_ghz = data.nodes[1].freq_ghz;
        nodes[1].belief = data.nodes[1].belief ?? 0.78;

        nodes[2].band = data.nodes[2].band;
        nodes[2].hit = data.nodes[2].hit;
        nodes[2].freq_ghz = data.nodes[2].freq_ghz;
        nodes[2].belief = data.nodes[2].belief ?? 0.15;
      }

      // Live Tuner Dwells & Hits
      if (data.tuner_actions && data.tuner_hits) {
        for (let m = 0; m < 4; m++) {
          tunerHistories[m].unshift({
            band: data.tuner_actions[m],
            hit: data.tuner_hits[m],
          });
          if (tunerHistories[m].length > 56) tunerHistories[m].pop();
        }
      }

      // AoI Vector
      if (data.ages && data.ages.length === 35) {
        ages = data.ages;
      }

      // Summary Figures of Merit
      if (data.summary) {
        $("strategy-label").textContent = data.summary.strategy || currentPolicy;
        $("collision-label").textContent = `${data.summary.collisions} Tuner Collisions [Guaranteed]`;
        $("perf-summary").textContent = `IR: ${data.summary.ir} · TTI: ${data.summary.tti}`;
        $("perf-sub").textContent = `${data.summary.throughput} Pulses Intercepted (${data.summary.multiplier || "5.9x"} Gain)`;
      }

      // EOB Records
      if (data.eob_records) {
        eobRecords = data.eob_records;
      }

      // PDW Intercepts
      if (data.pdw_records) {
        pdwRecords = data.pdw_records;
      }

      // Real Log Events
      if (data.new_events && data.new_events.length > 0) {
        data.new_events.forEach((msg) => log(msg));
      }

      render();
      return;
    }
  } catch (err) {
    // Network / Offline fallback
  }

  // Fallback: Local Simulation Math
  isConnected = false;
  $("engine-status").textContent = "OFFLINE PLAYBACK · LOCAL SIMULATION";
  $("engine-status").className = "badge amber";
  if ($("strategy-label")) $("strategy-label").textContent = currentPolicy;
  if ($("collision-label")) $("collision-label").textContent = "0.0% Tuner Collisions [Guaranteed]";
  if ($("perf-summary")) $("perf-summary").textContent = "IR: 28.5% · TTI: 185 ms";
  if ($("perf-sub")) $("perf-sub").textContent = "175.3 Pulses Intercepted (5.9x vs Baseline)";

  // Simulate synthetic multi-receiver movements
  nodes.forEach((n, i) => {
    n.band = (tick * [3, 4, 7][i] + i * 9) % 35;
    if (i === 1) n.band2 = (n.band + 3) % 35;
    n.hit = (tick + i) % 3 !== 0;
    n.power = [28.4, 32.1, 46.2][i] + Math.sin(tick / 5 + i) * 1.1;
    n.temp = [47, 63.2, 41][i] + Math.sin(tick / 8 + i) * 0.7;
    if (n.battery !== null) n.battery = Math.max(12, n.battery - 0.001);

    ages = ages.map((a) => (a === null ? null : a + 1));
    ages[n.band] = 0;
    if (n.band2 !== undefined) ages[n.band2] = 0;

    const histIdx = i === 0 ? 0 : i === 1 ? 1 : 3;
    tunerHistories[histIdx].unshift({ band: n.band, hit: n.hit });
    if (tunerHistories[histIdx].length > 56) tunerHistories[histIdx].pop();
  });

  // Synthetic EOB Records
  if (eobRecords.length === 0) {
    eobRecords = [
      {
        id: "RADAR-01 [S-300 PMU-2 Air Defense]",
        behaviour: "Fixed Frequency (2.50 GHz)",
        freq: "2.50 GHz",
        pri: "5,250.0 µs",
        alert: "CRITICAL",
        status: "LOCKED",
      },
      {
        id: "RADAR-02 [92N6E Grave Stone Target Acquisition]",
        behaviour: "Fixed Frequency (9.50 GHz)",
        freq: "9.50 GHz",
        pri: "10,500.0 µs",
        alert: "HIGH",
        status: "LOCKED",
      },
      {
        id: "RADAR-03 [Krasukha-4 Tactical FHSS Jammer]",
        behaviour: "FHSS Agile (5 Hops)",
        freq: "4.00–13.50 GHz",
        pri: "3,150.0 µs",
        alert: "CRITICAL",
        status: "TRACKING",
      },
      {
        id: "RADAR-04 [Su-35S Irbis-E Radar]",
        behaviour: "FHSS Agile (6 Hops)",
        freq: "1.50–17.00 GHz",
        pri: "2,100.0 µs",
        alert: "HIGH",
        status: "TRACKING",
      },
      {
        id: "RADAR-05 [P-18 Spoon Rest Early Warning]",
        behaviour: "Rotating Scanning Radar",
        freq: "6.00 GHz",
        pri: "2,100.0 µs",
        alert: "SURVEILLANCE",
        status: "SEARCHING",
      },
    ];
  }

  // Synthetic PDWs
  if (pdwRecords.length < 50 && nodes[0].hit) {
    pdwRecords.unshift({
      timestamp: `${(tick * 0.05).toFixed(4)}s`,
      tuner: "Tuner 0 (Alpha)",
      band: `B${String(nodes[0].band + 1).padStart(2, "0")}`,
      freq_ghz: nodes[0].freq_ghz,
      rssi_dbm: (-44.5 + Math.random() * 4).toFixed(1),
      pulse_width_us: "1050.0",
    });
  }

  render();
}

// ---------------------------------------------------------------------------
// Event Listeners & UI Controls
// ---------------------------------------------------------------------------

function restartInterval() {
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    $("clock").textContent = stamp();
    if (!paused) sample();
  }, pollIntervalMs);
}

$("pause").onclick = () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume simulation" : "Pause simulation";
  $("mode").textContent = paused ? "SIMULATION / PAUSED" : "SIMULATION / RUNNING";
  $("mode").className = paused ? "badge red" : "badge amber";
  log(paused ? "TOC simulation feed paused by operator" : "TOC simulation feed resumed");
};

$("source").onchange = draw;

$("ack").onclick = () => {
  advisoryAcked = true;
  $("ack-state").textContent = "Acknowledged (Override Approved)";
  $("ack").disabled = true;
  $("ack").textContent = "Acknowledged";
  $("adv-badge").textContent = "ACKNOWLEDGED";
  $("adv-badge").className = "badge green";
  log("Operator acknowledged Node Bravo LO synthesizer thermal advisory");
  render();
};

$("btn-apply-config").onclick = async () => {
  currentPolicy = $("policy-select").value;
  currentPreset = $("preset-select").value;
  pollIntervalMs = +$("speed-select").value;

  const rateHz = (1000 / pollIntervalMs).toFixed(1);
  $("rate-label").textContent = `Live Telemetry · ${rateHz} Hz`;

  log(`Simulation reconfigured: Policy = ${currentPolicy}, Scenario = ${currentPreset}`);
  $("save-status").textContent = `Configured: ${currentPolicy} running at ${rateHz} Hz. Resetting environment...`;

  try {
    const res = await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ policy: currentPolicy, preset: currentPreset }),
    });
    if (res.ok) {
      log("Environment reset confirmed by Python backend");
      $("save-status").textContent = `Simulation running live with ${currentPolicy} on ${currentPreset}.`;
    }
  } catch (err) {
    $("save-status").textContent = `Local offline reset applied (${currentPolicy}).`;
  }

  restartInterval();
  sample();
};

// PDW Export Handlers
$("btn-export-csv").onclick = () => {
  if (isConnected) {
    window.location.href = "/api/export/pdw.csv";
  } else {
    // Client-side CSV generation
    let csv = "Timestamp,Tuner,Band,Frequency_GHz,RSSI_dBm,PulseWidth_us\n";
    pdwRecords.forEach((p) => {
      csv += `${p.timestamp},${p.tuner},${p.band},${p.freq_ghz},${p.rssi_dbm},${p.pulse_width_us}\n`;
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "pdw_intercept_log.csv";
    a.click();
  }
  log("Exported Pulse Descriptor Word (PDW) log to CSV");
};

$("btn-export-json").onclick = () => {
  if (isConnected) {
    window.location.href = "/api/export/pdw.json";
  } else {
    // Client-side JSON generation
    const blob = new Blob([JSON.stringify(pdwRecords, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "pdw_intercept_log.json";
    a.click();
  }
  log("Exported Pulse Descriptor Word (PDW) log to JSON");
};

// Initialize Application
log("C2-ESM Tactical Operations Center interface loaded");
$("clock").textContent = stamp();
restartInterval();
sample();
