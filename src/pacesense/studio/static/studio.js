/* PaceSense Studio — interface. No framework, no build step; talks to /api/v1. */
"use strict";
(() => {
  // ---------------------------------------------------------------- helpers
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const NS = "http://www.w3.org/2000/svg";

  /** DOM builder. Text is always set as text (never parsed as HTML), so user input is safe. */
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
    return e;
  }
  function s(tag, attrs, ...kids) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const c of kids.flat()) if (c) e.append(c);
    return e;
  }
  const clear = (e) => { while (e.firstChild) e.firstChild.remove(); return e; };
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const storage = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* private mode */ } },
  };

  const fmt = {
    time(sec, withS = true) {
      if (sec == null || !isFinite(sec)) return "–";
      const t = Math.round(sec), hh = Math.floor(t / 3600), mm = Math.floor((t % 3600) / 60), ss = t % 60;
      return withS ? `${hh}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}` : `${hh}:${String(mm).padStart(2, "0")}`;
    },
    pace(sPerKm) {
      if (sPerKm == null || !isFinite(sPerKm)) return "–";
      const t = Math.round(sPerKm); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
    },
    signed(sec) {
      if (sec == null || !isFinite(sec)) return "–";
      const t = Math.round(Math.abs(sec)), sign = sec > 0.5 ? "+" : sec < -0.5 ? "−" : "±";
      return t >= 3600 ? `${sign}${fmt.time(t)}` : `${sign}${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
    },
    pct(x, d = 1) { return x == null ? "–" : `${(100 * x).toFixed(d)}%`; },
  };
  /** Charts are drawn at the container's real pixel width, so text stays at its true size on every screen. */
  function chartWidth(box, fallback = 760) {
    const card = box.closest(".card");
    const w = box.clientWidth || (card ? card.clientWidth - 36 : fallback);
    return Math.max(300, Math.round(w || fallback));
  }
  function niceStep(span, target = 4, steps = [5, 10, 15, 30, 60, 120, 180, 300, 600, 900, 1800, 3600]) {
    return steps.find((x) => span / x <= target) || steps[steps.length - 1];
  }
  const SEG_KM = [5, 5, 5, 5, 5, 5, 5, 5, 2.195];
  const SEG_END = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K", "Finish"];

  // ---------------------------------------------------------------- API
  class ApiError extends Error {
    constructor(p, status) { super(p.detail || p.title || `HTTP ${status}`); this.status = status; this.problem = p; }
  }
  async function api(method, path, body) {
    const res = await fetch(`/api/v1${path}`, {
      method, headers: body ? { "content-type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 204) return null;
    const data = await res.json().catch(() => ({ title: res.statusText }));
    if (!res.ok) throw new ApiError(data, res.status);
    return data;
  }

  // ---------------------------------------------------------------- state
  const app = {
    sid: null, st: null, sim: null, busy: false, reactions: {}, lastCard: null, meta: null,
    hoverRace: null,
  };

  // ---------------------------------------------------------------- toasts + tooltip
  function toast(msg, kind = "") {
    const t = h("div", { class: `toast ${kind}`, text: msg });
    $("#toasts").append(t);
    setTimeout(() => t.remove(), kind === "error" ? 6000 : 3200);
  }
  const tip = $("#tooltip");
  function showTip(evt, ...lines) {
    clear(tip).append(...lines.map((l) => h("div", null, l)));
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - 14;
    tip.style.left = `${x}px`; tip.style.top = `${y}px`;
  }
  const hideTip = () => { tip.hidden = true; };
  const tipLine = (label, value) => h("span", null, `${label} `, h("b", { text: value }));

  // ---------------------------------------------------------------- session lifecycle
  async function openRunner(runnerId = "demo", holdout = null, condition = null) {
    condition = condition || storage.get("pacesense.condition") || "ask";
    setBusy(true);
    try {
      const r = await api("POST", "/sessions", { runner_id: runnerId, holdout, coach_id: coachId(), condition });
      if (app.sid) api("DELETE", `/sessions/${app.sid}`).catch(() => {});
      app.sid = r.id; app.reactions = {}; app.lastCard = null; app.sim = null;
      storage.set("pacesense.sid", r.id);
      clear($("#feed"));
      render(r.state);
      renderFeedFromTurns(r.state);
      adapt(r.state.adapt, { fresh: true });
      runSim();
      return true;
    } catch (e) { toast(e.message, "error"); return false; } finally { setBusy(false); }
  }

  async function resume() {
    const sid = storage.get("pacesense.sid");
    if (sid) {
      try {
        const st = await api("GET", `/sessions/${sid}`);
        if (st.session.coach_id !== coachId()) throw new Error("other coach");
        app.sid = sid; render(st); renderFeedFromTurns(st); adapt(st.adapt, { fresh: true, resumed: true }); runSim(); return;
      } catch { storage.set("pacesense.sid", null); }
    }
    await openRunner("demo");
  }

  function setBusy(b) {
    app.busy = b;
    $("#main").setAttribute("aria-busy", String(b));
    $("#send-btn").classList.toggle("is-busy", b);
  }

  // ---------------------------------------------------------------- render: everything
  function render(st) {
    const prev = app.st && app.st.races ? app.st.races.map((r) => r.weight) : null;
    app.st = st;
    renderHeader(st); renderKpis(st); renderRaces(st); animateWeights(prev, st);
    renderPacing(st); renderSimilar(st); renderInsights(st); renderSuggestions();
    $("#undo-btn").disabled = !st.session.can_undo;
    $("#ai-pill").hidden = !st.session.ai;
  }

  function renderHeader(st) {
    const r = st.runner, demo = st.mode === "demo";
    $("#runner-name").textContent = demo ? `Demo runner · ${st.session.source_runner}` : r.name;
    const g = r.gender === "F" ? "Female" : r.gender === "M" ? "Male" : "";
    $("#runner-meta").textContent = [g, r.age ? `${r.age} yrs` : "", `${st.races.length} recent races`].filter(Boolean).join(" · ");
    $("#runner-avatar").textContent = (r.gender || "R") + (r.age ? String(r.age) : "");
    const pill = $("#mode-pill");
    pill.hidden = !st.holdout;
    if (st.holdout) pill.textContent = st.holdout.revealed ? `Revealed: ${st.holdout.label}` : "Best race hidden";
    document.title = `${demo ? "Demo runner" : r.name} · PaceSense Studio`;
  }

  // ---------------------------------------------------------------- KPIs
  function renderKpis(st) {
    const p = st.prediction;
    $("#k-target").textContent = p.finish;
    const d = p.finish_s - p.initial_s, sub = $("#k-target-sub");
    clear(sub);
    if (Math.abs(d) < 0.5) sub.append(`Equal weights · ${p.n_cases} similar cases`);
    else sub.append("vs equal weights ", h("span", { class: d > 0 ? "delta-up" : "delta-down", text: fmt.signed(d) }),
      ` (${p.initial})`);

    $("#k-range").textContent = p.low_s ? `${fmt.time(p.low_s)} – ${fmt.time(p.high_s)}` : "–";
    drawRangeBar(st);

    // accuracy
    const acc = $("#k-acc"), accSub = $("#k-acc-sub"), ho = st.holdout;
    clear(acc); clear(accSub);
    if (!ho) {
      acc.append(h("span", { class: "muted", text: "No hidden race" }));
      accSub.append("Open the demo runner, or tick “hide the best race”, to test accuracy.");
    } else if (!ho.revealed) {
      acc.append(h("span", { text: ho.label }), " ", h("span", { class: "muted", text: "is hidden" }));
      const last = app.lastCard && app.lastCard.accuracy;
      accSub.append(last ? h("span", { class: last.closer ? "delta-down" : last.same ? "" : "delta-up", text: `Last change: ${last.text}.` })
        : "Steer the model, then compare.");
      accSub.append(h("div", null, h("button", { class: "btn small reveal", type: "button", onclick: reveal }, "Reveal the real result")));
    } else {
      const better = ho.error_pct <= ho.equal_weights_error_pct + 1e-9;
      acc.append(h("div", { class: "acc-line" }, h("span", { class: `acc-big ${better ? "delta-down" : "delta-up"}`, text: `${ho.error_pct.toFixed(2)}%` }),
        h("span", { class: "muted", text: "off" })));
      accSub.append(`Real: ${fmt.time(ho.truth_s)} (${ho.label}). Equal weights were ${ho.equal_weights_error_pct.toFixed(2)}% off`
        + (better ? ", so your changes helped." : ", so your changes did not help."));
    }

    // steering
    const sv = $("#k-steer"); clear(sv);
    const stg = st.steering;
    if (stg.index == null) {
      sv.append(h("span", { class: "muted", text: "Add your estimate" }));
    } else {
      const pctv = clamp(stg.index, -0.5, 1.5);
      sv.append(h("span", { class: "mono", text: `${Math.round(100 * stg.index)}%` }), " ",
        h("span", { class: "muted", text: `towards your ${fmt.time(stg.coach_estimate_s, false)}` }));
      const m = s("svg", { class: "steer-meter", viewBox: "0 0 200 10", preserveAspectRatio: "none", role: "img",
        "aria-label": stg.label });
      m.append(s("rect", { x: 0, y: 3, width: 200, height: 4, rx: 2, fill: "var(--line)" }),
        s("rect", { x: 0, y: 3, width: 200 * clamp(pctv, 0, 1), height: 4, rx: 2, fill: "var(--know)" }),
        s("circle", { cx: 200 * clamp(pctv, 0, 1), cy: 5, r: 4.5, fill: "var(--surface)", stroke: "var(--know)", "stroke-width": 2 }));
      sv.append(m);
      sv.title = stg.label;
    }
  }

  function drawRangeBar(st) {
    const svg = clear($("#k-range-bar")), p = st.prediction;
    if (!p.low_s) return;
    const W = Math.max(160, Math.round(svg.getBoundingClientRect().width || 300));
    svg.setAttribute("viewBox", `0 0 ${W} 34`);
    const ho = st.holdout, pts = [p.low_s, p.high_s, p.finish_s, p.initial_s];
    if (ho && ho.revealed) pts.push(ho.truth_s);
    const lo = Math.min(...pts), hi = Math.max(...pts), pad = (hi - lo) * 0.12 + 30;
    const x = (v) => 6 + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - 12);
    svg.append(
      s("rect", { x: 0.5, y: 9, width: W - 1, height: 6, rx: 3, fill: "var(--surface-2)", stroke: "var(--line)" }),
      s("rect", { x: x(p.low_s), y: 9, width: x(p.high_s) - x(p.low_s), height: 6, rx: 3, fill: "var(--accent)", opacity: 0.28 }),
      s("line", { x1: x(p.initial_s), x2: x(p.initial_s), y1: 5, y2: 19, stroke: "var(--muted)", "stroke-width": 1.5, "stroke-dasharray": "2 2" }),
      s("circle", { cx: x(p.finish_s), cy: 12, r: 5.5, fill: "var(--accent)", stroke: "var(--surface)", "stroke-width": 2 }),
      s("text", { x: x(p.low_s), y: 31, "text-anchor": "middle", text: fmt.time(p.low_s, false) }),
      s("text", { x: x(p.high_s), y: 31, "text-anchor": "middle", text: fmt.time(p.high_s, false) }),
    );
    if (ho && ho.revealed) {
      const tx = x(ho.truth_s);
      svg.append(s("path", { d: `M${tx} 4 l5 8 l-5 8 l-5 -8z`, fill: "var(--good)", stroke: "var(--surface)", "stroke-width": 1.5 }));
    }
    svg.onmousemove = (e) => showTip(e, tipLine("Target", p.finish), tipLine("Likely", `${fmt.time(p.low_s)} – ${fmt.time(p.high_s)}`),
      tipLine("Equal weights", p.initial), ...(ho && ho.revealed ? [tipLine("Real result", fmt.time(ho.truth_s))] : []));
    svg.onmouseleave = hideTip;
  }

  // ---------------------------------------------------------------- races table
  function evidenceChip(ev) {
    if (ev.direction > 0) return h("span", { class: "chip warn", text: "slower than usual" });
    if (ev.direction < 0) return h("span", { class: "chip ok", text: "faster than usual" });
    return h("span", { class: "chip neutral", text: "typical" });
  }

  function evidenceText(r) {
    const med = r.evidence.others_median_s;
    if (med == null) return r.evidence.label;
    const diff = r.finish_s - med, min = Math.round(Math.abs(diff) / 60);
    if (r.evidence.direction === 0) return `Within her usual spread (others: ${fmt.time(med, false)})`;
    return `${min} min ${diff > 0 ? "slower" : "faster"} than her other races (${fmt.time(med, false)})`;
  }

  function renderRaces(st) {
    const body = clear($("#race-rows"));
    const latest = Math.max(...st.races.map((r) => r.year));
    const maxPull = Math.max(30, ...st.races.map((r) => Math.abs(r.influence.pull_s || 0)));
    st.races.forEach((r, i) => {
      const slider = h("input", { type: "range", min: 0, max: 1, step: 0.05, value: r.weight,
        "aria-label": `Weight for ${r.event} ${r.year}` });
      const out = h("output", { class: r.weight === 0 ? "zero" : "", text: r.weight.toFixed(2) });
      const setFill = () => slider.style.setProperty("--fill", `${slider.value * 100}%`);
      setFill();
      slider.addEventListener("input", () => { out.textContent = Number(slider.value).toFixed(2); setFill(); });
      slider.addEventListener("change", () => setWeight(i, Number(slider.value)));

      const pull = r.influence.pull_s || 0, frac = (Math.abs(pull) / maxPull) * 50;
      const fill = h("span", { class: `pull-fill ${pull < 0 ? "faster" : "slower"}` });
      fill.style.width = `${frac}%`;
      fill.style.left = pull < 0 ? `${50 - frac}%` : "50%";
      const bar = h("div", { class: "pull-box", role: "img", "aria-label": r.weight === 0 ? "ignored" : `${fmt.signed(pull)} on the target`,
        title: pull < 0 ? "Pulls the target faster" : "Pulls the target slower" },
        h("div", { class: "pull-track" }, fill),
        h("span", { class: `pull-val mono ${r.weight === 0 ? "ignored" : ""}`, text: r.weight === 0 ? "ignored" : `${fmt.signed(pull)}` }));

      const ago = latest - r.year;
      const tr = h("tr", { "data-i": i },
        h("td", { class: "race-name" }, h("b", { text: `${r.event} ${r.year}` }),
          h("small", { text: ago === 0 ? "most recent" : `${ago} yr${ago > 1 ? "s" : ""} before the latest` })),
        h("td", { class: "num mono", text: r.finish }),
        h("td", null, evidenceChip(r.evidence), h("div", { class: "ev-text", text: evidenceText(r) })),
        h("td", null, h("div", { class: "weight" }, slider, out)),
        h("td", { class: "pull" }, bar));
      tr.addEventListener("mouseenter", () => highlightRace(i));
      tr.addEventListener("mouseleave", () => highlightRace(null));
      if (app.changed && app.changed.has(i)) tr.classList.add("changed");
      if ((app.spotlight || []).includes(i)) tr.classList.add("spot");
      body.append(tr);
    });
    app.changed = null;
    const sub = $("#races-sub");
    sub.textContent = `How much each race counts · ${st.prediction.n_cases} similar cases`
      + (st.half_life_years ? ` · recent races count more (half-life ${st.half_life_years} yrs)` : "");
  }

  /** When the voice assistant (or anything else) changes a weight, the slider visibly travels to it. */
  function animateSlider(slider, out, from, to, ms = 650) {
    if (from == null || Math.abs(from - to) < 1e-9 || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const t0 = performance.now();
    slider.classList.add("moving");
    const step = (now) => {
      const k = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - k, 3), v = from + (to - from) * e;
      slider.value = v; slider.style.setProperty("--fill", `${v * 100}%`);
      if (out) out.textContent = v.toFixed(2);
      if (k < 1) requestAnimationFrame(step); else { slider.value = to; slider.classList.remove("moving"); }
    };
    requestAnimationFrame(step);
  }
  function animateWeights(prev, st) {
    if (!prev) return;
    $$("#race-rows tr[data-i]").forEach((tr) => {
      const i = Number(tr.dataset.i), sl = $("input[type=range]", tr), out = $("output", tr);
      if (sl && st.races[i] && prev[i] != null) animateSlider(sl, out, prev[i], st.races[i].weight);
    });
  }

  function highlightRace(i) {
    app.hoverRace = i;
    $$("#race-rows tr").forEach((tr) => tr.classList.toggle("hl", Number(tr.dataset.i) === i));
    $$("#pacing-chart [data-race]").forEach((p) => {
      const on = i != null && Number(p.dataset.race) === i;
      p.setAttribute("stroke", on ? "var(--ink)" : "var(--past)");
      p.setAttribute("stroke-width", on ? 2.2 : 1.4);
      if (on) p.parentNode.append(p);
    });
  }

  // ---------------------------------------------------------------- pacing chart
  function renderPacing(st) {
    const box = clear($("#pacing-chart"));
    const plan = st.plan.paces, races = st.race_paces || [];
    const W = chartWidth(box), narrow = W < 520, H = narrow ? 210 : 250, m = { l: 46, r: 14, t: 14, b: 30 };
    const all = plan.concat(...races.filter(Boolean));
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * 0.08 + 3; lo -= pad; hi += pad;
    const x = (i) => m.l + (i / (plan.length - 1)) * (W - m.l - m.r);
    const y = (v) => m.t + ((v - lo) / (hi - lo)) * (H - m.t - m.b);           // faster (smaller) is higher
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Pace per 5 km: the plan and her past races" });

    const grid = s("g", { class: "grid" }), axis = s("g", { class: "axis" });
    const step = (hi - lo) > 60 ? 15 : (hi - lo) > 30 ? 10 : 5;
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      grid.append(s("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
      axis.append(s("text", { x: m.l - 8, y: y(v) + 3.5, "text-anchor": "end", text: fmt.pace(v) }));
    }
    st.plan.labels.forEach((l, i) => (!narrow || i % 2 === 0) &&
      axis.append(s("text", { x: x(i), y: H - m.b + 18, "text-anchor": i === plan.length - 1 ? "end" : i === 0 ? "start" : "middle",
        text: i === plan.length - 1 ? `${l} km` : l })));
    axis.append(s("text", { class: "axis-title", x: 2, y: m.t - 2, text: "min/km" }));
    svg.append(grid, axis);

    const line = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    const pastG = s("g");
    races.forEach((rp, i) => {
      if (!rp) return;
      const w = st.races[i] ? st.races[i].weight : 0.5;
      pastG.append(s("path", { d: line(rp), fill: "none", stroke: "var(--past)", "stroke-width": 1.4,
        "stroke-dasharray": w === 0 ? "3 4" : null, opacity: 0.35 + 0.65 * w, "data-race": i }));
    });
    svg.append(pastG);
    svg.append(s("path", { d: `${line(plan)}`, fill: "none", stroke: "var(--accent)", "stroke-width": 2.8, "stroke-linejoin": "round" }));
    plan.forEach((v, i) => svg.append(s("circle", { cx: x(i), cy: y(v), r: 3.6, fill: "var(--surface)", stroke: "var(--accent)", "stroke-width": 2 })));

    // hover guide
    const guide = s("line", { y1: m.t, y2: H - m.b, stroke: "var(--line-strong)", "stroke-dasharray": "3 3", visibility: "hidden" });
    svg.append(guide);
    const hit = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
    hit.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W;
      const i = clamp(Math.round(((px - m.l) / (W - m.l - m.r)) * (plan.length - 1)), 0, plan.length - 1);
      guide.setAttribute("x1", x(i)); guide.setAttribute("x2", x(i)); guide.setAttribute("visibility", "visible");
      const rows = [h("span", null, `${st.plan.labels[i]} km`), tipLine("Plan", `${fmt.pace(plan[i])}/km`)];
      races.forEach((rp, j) => rp && rows.push(tipLine(`${st.races[j].event} ${st.races[j].year}`, `${fmt.pace(rp[j === -1 ? 0 : i])}/km`)));
      showTip(e, ...rows);
    });
    hit.addEventListener("mouseleave", () => { guide.setAttribute("visibility", "hidden"); hideTip(); });
    svg.append(hit);
    box.append(h("div", { class: "chart-legend" },
      h("span", { class: "lg plan" }, "Plan (from the model)"),
      h("span", { class: "lg past" }, "Her past races · darker = more weight"),
      h("span", { class: "lg ignored" }, "ignored"),
      h("span", { class: "muted", text: "faster is higher · hover a race row to pick it out" })), svg);

    // split strip
    const strip = clear($("#split-strip"));
    let cum = 0;
    plan.forEach((p, i) => {
      cum += p * SEG_KM[i];
      strip.append(h("div", { title: `${fmt.pace(p)}/km for ${st.plan.labels[i]} km` }, h("small", { text: SEG_END[i] }), fmt.time(cum)));
    });
    const chk = st.plan.check || {};
    $("#analysis-sub").textContent = chk.note ? `Plan check: ${chk.note}` : "";
  }

  // ---------------------------------------------------------------- similar runners
  function renderSimilar(st) {
    const box = clear($("#similar-chart"));
    const rows = (st.similar || []).map((sm, i) => sm && { ...sm, race: st.races[i] }).filter(Boolean);
    if (!rows.length) { box.append(h("p", { class: "muted", text: "No similar runners found for these races." })); return; }
    const p = st.prediction;
    const vals = rows.flatMap((r) => r.finishes_s).concat([p.low_s, p.high_s, p.finish_s].filter(Boolean));
    const qs = vals.slice().sort((a, b) => a - b);
    let lo = qs[Math.floor(qs.length * 0.01)], hi = qs[Math.ceil(qs.length * 0.99) - 1];
    lo = Math.min(lo, p.low_s || lo) - 120; hi = Math.max(hi, p.high_s || hi) + 120;
    const W = chartWidth(box), narrow = W < 520, rowH = 40, m = { l: narrow ? 92 : 150, r: 14, t: 24, b: 34 },
      H = m.t + rows.length * rowH + m.b;
    const x = (v) => m.l + ((clamp(v, lo, hi) - lo) / (hi - lo)) * (W - m.l - m.r);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "What similar runners went on to run, for each of her races" });
    if (p.low_s) svg.append(s("rect", { x: x(p.low_s), y: m.t - 8, width: x(p.high_s) - x(p.low_s), height: rows.length * rowH + 8, fill: "var(--band)" }));
    const axis = s("g", { class: "axis" }), grid = s("g", { class: "grid" });
    const tick = niceStep(hi - lo, narrow ? 3 : 6, [300, 600, 900, 1800, 3600]);
    for (let v = Math.ceil(lo / tick) * tick; v <= hi; v += tick) {
      grid.append(s("line", { x1: x(v), x2: x(v), y1: m.t - 8, y2: H - m.b }));
      axis.append(s("text", { x: x(v), y: H - m.b + 16, "text-anchor": "middle", text: fmt.time(v, false) }));
    }
    axis.append(s("text", { class: "axis-title", x: W - m.r, y: H - 2, "text-anchor": "end", text: "what they ran next" }));
    svg.append(grid, axis);
    rows.forEach((r, k) => {
      const cy = m.t + k * rowH + rowH / 2;
      const w = r.race.weight;
      svg.append(s("text", { x: 0, y: cy - 2, class: "rowlbl", text: narrow ? `${r.race.event.slice(0, 9)} ’${String(r.race.year).slice(2)}` : `${r.race.event} ${r.race.year}` }),
        s("text", { x: 0, y: cy + 13, class: "lbl", text: narrow ? `w ${w.toFixed(2)}` : `${r.n} runners · weight ${w.toFixed(2)}` }));
      const g = s("g", { opacity: w === 0 ? 0.3 : 1 });
      r.finishes_s.forEach((v, j) => g.append(s("circle", { cx: x(v), cy: cy + ((j % 3) - 1) * 4, r: 3.2, fill: "var(--accent)", opacity: 0.45 })));
      g.append(s("line", { x1: x(r.median_s), x2: x(r.median_s), y1: cy - 11, y2: cy + 11, stroke: "var(--ink)", "stroke-width": 2 }));
      const hit = s("rect", { x: m.l, y: cy - rowH / 2, width: W - m.l - m.r, height: rowH, fill: "transparent" });
      hit.addEventListener("mousemove", (e) => showTip(e, h("span", null, `Runners who ran ${r.race.finish} like her ${r.race.event} ${r.race.year}`),
        tipLine("went on to run, median", fmt.time(r.median_s)),
        tipLine("middle half", `${fmt.time(r.finishes_s[Math.floor(r.n * 0.25)])} – ${fmt.time(r.finishes_s[Math.floor(r.n * 0.75)])}`)));
      hit.addEventListener("mouseleave", hideTip);
      svg.append(g, hit);
    });
    svg.append(s("line", { x1: x(p.finish_s), x2: x(p.finish_s), y1: m.t - 12, y2: H - m.b, stroke: "var(--accent)", "stroke-width": 2 }),
      s("text", { x: x(p.finish_s), y: m.t - 14, "text-anchor": "middle", class: "lbl", fill: "var(--accent-ink)", text: `target ${p.finish}` }));
    box.append(svg, h("p", { class: "footnote", text: "Each dot is a runner who ran about the same time as her race, and where they finished their next marathon. Black tick: median. Shaded: her likely range." }));
  }

  function renderInsights(st) {
    const ul = clear($("#insights"));
    (st.insights || []).forEach((t) => ul.append(h("li", { text: t })));
    const sc = clear($("#scenarios"));
    (st.scenarios || []).forEach((x) => sc.append(h("div", { class: "scenario" }, h("small", { text: x.name }), h("b", { text: x.finish }))));
  }

  // ---------------------------------------------------------------- simulator
  function simParams() {
    return { temp_c: Number($("#sim-temp").value), wind_kmh: Number($("#sim-wind").value),
      profile: $("#sim-profile input:checked").value };
  }
  function syncSimLabels() {
    for (const [id, out, unit] of [["#sim-temp", "#temp-out", " °C"], ["#sim-wind", "#wind-out", " km/h"]]) {
      const el = $(id); $(out).textContent = el.value + unit;
      el.style.setProperty("--fill", `${((el.value - el.min) / (el.max - el.min)) * 100}%`);
    }
  }
  const runSim = debounce(async () => {
    if (!app.sid) return;
    try { app.sim = await api("POST", `/sessions/${app.sid}/simulations`, simParams()); renderSim(app.sim); }
    catch (e) { toast(e.message, "error"); }
  }, 180);
  function setSim({ temp_c, wind_kmh, profile }) {
    $("#sim-temp").value = Math.round(temp_c); $("#sim-wind").value = Math.round(wind_kmh);
    const r = $(`#sim-profile input[value="${profile}"]`); if (r) r.checked = true;
    syncSimLabels();
  }

  function renderSim(sim) {
    const sm = sim.summary, stats = clear($("#sim-stats"));
    const stat = (label, value, cls = "") => h("div", { class: "stat" }, h("small", { text: label }), h("b", { class: cls, text: value }));
    stats.append(stat("Simulated finish", fmt.time(sm.sim_finish_s)),
      stat("vs plan", fmt.signed(sm.delta_s), sm.delta_s > 1 ? "delta-up" : sm.delta_s < -1 ? "delta-down" : ""),
      stat("In likely range?", sm.inside_range ? "Yes" : "No", sm.inside_range ? "delta-down" : "delta-up"),
      stat("A minute behind at", sm.first_minute_behind_km == null ? "never" : `${sm.first_minute_behind_km} km`));
    $("#sim-note").textContent = `${sm.profile_text}. Conditions add ${sm.conditions_pct.toFixed(1)}%. ${sim.assumptions}`;

    const box = clear($("#sim-chart")), fr = sim.frames;
    const W = chartWidth(box, 700), narrow = W < 520, H = narrow ? 190 : 220, m = { l: 50, r: 14, t: 18, b: 30 };
    const d = fr.map((f) => ({ km: f.km, gap: f.sim_s - f.plan_s, lo: f.low_s - f.plan_s, hi: f.high_s - f.plan_s, f }));
    const ext = Math.max(60, ...d.map((p) => Math.abs(p.gap)), ...d.map((p) => Math.abs(p.lo)), ...d.map((p) => Math.abs(p.hi))) * 1.1;
    const x = (km) => m.l + (km / 42.195) * (W - m.l - m.r);
    const y = (g) => m.t + ((g + ext) / (2 * ext)) * (H - m.t - m.b);           // behind = down
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Time ahead or behind the plan along the course" });
    const grid = s("g", { class: "grid" }), axis = s("g", { class: "axis" });
    for (const km of narrow ? [0, 10, 20, 30, 40] : [0, 5, 10, 15, 20, 25, 30, 35, 40]) {
      grid.append(s("line", { x1: x(km), x2: x(km), y1: m.t, y2: H - m.b }));
      axis.append(s("text", { x: x(km), y: H - m.b + 16, "text-anchor": "middle", text: km }));
    }
    const step = niceStep(ext, 2, [30, 60, 120, 180, 300, 600, 900, 1800]);
    for (let g = -Math.floor(ext / step) * step; g <= ext; g += step) {
      if (g === 0) continue;
      axis.append(s("text", { x: m.l - 8, y: y(g) + 3.5, "text-anchor": "end", text: fmt.signed(g) }));
    }
    axis.append(s("text", { class: "axis-title", x: m.l - 8, y: m.t - 4, "text-anchor": "end", text: "ahead" }),
      s("text", { class: "axis-title", x: m.l - 8, y: H - m.b + 2, "text-anchor": "end", text: "" }),
      s("text", { class: "axis-title", x: W - m.r, y: H - 2, "text-anchor": "end", text: "km" }));
    svg.append(grid, axis);
    const band = d.map((p, i) => `${i ? "L" : "M"}${x(p.km)},${y(p.lo)}`).join("") +
      d.slice().reverse().map((p) => `L${x(p.km)},${y(p.hi)}`).join("") + "Z";
    svg.append(s("path", { d: band, fill: "var(--band)" }),
      s("line", { class: "zero", x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), "stroke-width": 1.5 }),
      s("text", { x: W - m.r, y: y(0) - 5, "text-anchor": "end", class: "lbl", text: "on plan" }),
      s("text", { x: m.l + 6, y: H - m.b - 6, class: "lbl", text: "behind ↓" }));
    const col = sm.delta_s > 60 ? "var(--bad)" : sm.delta_s > 1 ? "var(--warn)" : "var(--good)";
    svg.append(s("path", { d: d.map((p, i) => `${i ? "L" : "M"}${x(p.km).toFixed(1)},${y(p.gap).toFixed(1)}`).join(""),
      fill: "none", stroke: col, "stroke-width": 2.6, "stroke-linejoin": "round" }));
    const last = d[d.length - 1];
    svg.append(s("circle", { cx: x(last.km), cy: y(last.gap), r: 4.5, fill: col, stroke: "var(--surface)", "stroke-width": 2 }));
    const dot = s("circle", { r: 4, fill: "var(--surface)", stroke: col, "stroke-width": 2, visibility: "hidden" });
    const hit = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
    hit.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect(), km = ((((e.clientX - r.left) / r.width) * W - m.l) / (W - m.l - m.r)) * 42.195;
      const p = d.reduce((a, b) => (Math.abs(b.km - km) < Math.abs(a.km - km) ? b : a));
      dot.setAttribute("cx", x(p.km)); dot.setAttribute("cy", y(p.gap)); dot.setAttribute("visibility", "visible");
      showTip(e, h("span", null, `${p.km.toFixed(1)} km`), tipLine("Clock", fmt.time(p.f.sim_s)),
        tipLine(p.gap > 0 ? "Behind plan" : "Ahead of plan", fmt.signed(Math.abs(p.gap))), ...(p.f.cue ? [h("span", { class: "muted", text: p.f.cue })] : []));
    });
    hit.addEventListener("mouseleave", () => { dot.setAttribute("visibility", "hidden"); hideTip(); });
    svg.append(dot, hit);
    box.append(svg);
  }

  // ---------------------------------------------------------------- coach notes feed
  const feed = $("#feed");
  function scrollFeed() { requestAnimationFrame(() => { feed.scrollTop = feed.scrollHeight; }); }
  function emptyFeed() {
    if (feed.children.length) return;
    const yrs = app.st ? app.st.races.map((r) => r.year) : [];
    feed.append(h("li", { class: "empty" }, h("b", { text: "Tell PaceSense what you know" }),
      "Explain an unusual race (“", yrs.length ? `${yrs[1] || yrs[0]} was a hot day` : "2016 was a hot day",
      "”), say which races to trust, or give your own estimate. Every change is checked against her times."));
  }
  function addSaid(text, source) {
    const empty = $(".empty", feed); if (empty) empty.remove();
    const manual = source === "manual" || source === "undo";
    feed.append(h("li", { class: `said ${manual ? "manual" : ""}` }, text,
      h("small", { text: source === "voice" ? "spoken" : manual ? (source === "undo" ? "undo" : "slider") : "typed" })));
    scrollFeed();
  }
  function addReply(text) { feed.append(h("li", { class: "reply", text })); scrollFeed(); }

  const VERDICT = { supported: ["ok", "matches the data"], your_knowledge: ["know", "your knowledge"],
    against_data: ["warn", "against the data"], undo: ["neutral", "undone"], no_change: ["neutral", "no change"] };

  function addCard(card) {
    app.lastCard = card;
    const items = h("div", { class: "fcard-items" });
    for (const it of card.items) {
      const [cls, label] = VERDICT[it.verdict] || ["neutral", it.verdict];
      items.append(h("div", { class: "fitem" },
        h("div", { class: "fitem-top" }, h("b", { text: it.race }),
          h("span", { class: "wchange", text: `${it.old.toFixed(2)} → ${it.new.toFixed(2)}` }),
          h("span", { class: `chip ${cls}`, text: label })),
        h("p", { text: it.evidence }),
        inlineSlider(it)));
    }
    const li = h("li", { class: "fcard" }, items);
    if (card.notes && card.notes.length) li.append(h("ul", { class: "notes-list" }, card.notes.map((n) => h("li", { text: n }))));
    if (card.accuracy) {
      const a = card.accuracy, cls = a.closer ? "closer" : a.same ? "same" : "further";
      li.append(h("div", { class: `acc-note ${cls}`, text: `Hidden-race check: ${a.text}.` }));
    }
    const react = h("div", { class: "react", role: "group", "aria-label": "Does this make sense?" });
    for (const [v, label] of [["agree", "Makes sense"], ["unsure", "Not sure"], ["disagree", "Disagree"]]) {
      const b = h("button", { type: "button", "aria-pressed": "false", text: label });
      b.addEventListener("click", () => sendReaction(card.n, v, react, b));
      react.append(b);
    }
    li.append(h("div", { class: "fcard-foot" },
      h("span", null, "Target ", h("span", { class: "mono", text: card.target }), " ",
        card.target_delta ? h("span", { class: card.target_delta.startsWith("+") ? "delta-up" : "delta-down", text: `(${card.target_delta})` }) : ""),
      react));
    feed.append(li); scrollFeed();
    renderSuggestions(card.suggestions);
  }

  /** The slider for this race, inside the conversation: it moves to the new weight, and can be dragged right here. */
  function inlineSlider(it) {
    if (it.index == null || !app.st || !app.st.races[it.index]) return "";
    const sl = h("input", { type: "range", min: 0, max: 1, step: 0.05, value: it.old, "aria-label": `Weight for ${it.race}` });
    const out = h("output", { class: "mono", text: Number(it.old).toFixed(2) });
    sl.style.setProperty("--fill", `${it.old * 100}%`);
    sl.addEventListener("input", () => { out.textContent = Number(sl.value).toFixed(2); sl.style.setProperty("--fill", `${sl.value * 100}%`); });
    sl.addEventListener("change", () => setWeight(it.index, Number(sl.value)));
    requestAnimationFrame(() => animateSlider(sl, out, it.old, it.new, 800));
    return h("div", { class: "inline-weight" }, h("small", { text: "weight" }), sl, out);
  }

  function renderFeedFromTurns(st) {
    clear(feed);
    for (const t of st.turns || []) {
      addSaid(t.text, t.source);
      if (t.applied && t.applied.length) {
        addCard({ n: t.n, items: t.applied.map((a) => ({ ...a })), notes: t.notes, target: fmt.time(t.finish_after_s),
          target_delta: fmt.signed(t.finish_after_s - t.finish_before_s), suggestions: [] });
      } else if (t.notes && t.notes.length) addReply(t.notes.join(" "));
    }
    emptyFeed(); renderSuggestions();
  }

  async function sendReaction(turn, value, group, btn) {
    try {
      const res = await api("POST", `/sessions/${app.sid}/reactions`, { turn, value });
      if (group) $$("button", group).forEach((b) => { b.setAttribute("aria-pressed", String(b === btn)); b.disabled = true; });
      adapt(res.adapt);
    } catch (e) { toast(e.message, "error"); }
  }

  function renderSuggestions(list) {
    const box = clear($("#suggestions"));
    let items = list && list.length ? list : null;
    if (!items && app.st) {
      const slow = app.st.races.find((r) => r.evidence.direction > 0) || app.st.races[0];
      items = [`${slow.year} was a hot day`, "trust the recent races more", "I think she can run "
        + fmt.time(app.st.prediction.finish_s - 240, false), "simulate 25 degrees", "what does the likely range mean?"];
    }
    (items || []).forEach((t) => box.append(h("button", { type: "button", class: "chip-btn", text: t, onclick: () => send(t, "typed") })));
  }

  // ---------------------------------------------------------------- actions
  /** Messages are processed one at a time, in order (voice can produce several quickly). */
  let queue = Promise.resolve();
  function send(text, source = "typed", signals = null) {
    queue = queue.then(() => sendNow(text, source, signals)).catch(() => {});
    return queue;
  }
  /** Everything a turn can bring back: new state (sliders animate), a card or reply, UI actions, adaptations. */
  async function applyTurn(r) {
    app.changed = new Set((r.card && r.card.items || []).map((it) => r.state.races.findIndex((x) => `${x.event} ${x.year}` === it.race)));
    render(r.state);
    if (r.choice) feed.append(h("li", { class: "uinote choice-note" }, h("span", { text: `You chose: ${r.choice.label}` })));
    if (r.card && r.card.items.length) addCard(r.card);
    // A feedback card explains the change visually; keep the assistant's spoken/text reply too.
    // Only runner switches defer their reply until the new session has opened successfully.
    if (r.reply && !(r.actions || []).some((a) => a.type === "runner")) addReply(r.reply);
    const sessionChanged = await handleActions(r.actions || [], r.simulation);
    if (!sessionChanged) adapt(r.adapt);
    if (!(r.simulation) && !sessionChanged) runSim();
  }
  async function sendNow(text, source, signals) {
    text = text.trim();
    if (!text || !app.sid) return;
    if (/^undo$/i.test(text) && !(app.pending)) return undo();
    addSaid(text, source);
    const typing = h("li", { class: "typing", text: "Checking her data" }); feed.append(typing); scrollFeed();
    setBusy(true);
    try {
      const r = await api("POST", `/sessions/${app.sid}/commands`, { text, source, ...(signals ? { voice: signals } : {}) });
      typing.remove();
      await applyTurn(r);
    } catch (e) {
      typing.remove();
      if (e.status === 404) { toast("That session expired; opening the runner again."); return openRunner("demo"); }
      addReply(`Sorry, that didn't work: ${e.message}`);
    } finally { setBusy(false); }
  }

  const PANEL = { races: "#panel-races", prediction: ".kpis", pacing: "#panel-analysis", similar: "#panel-analysis",
    simulator: "#panel-sim", feedback: "#panel-notes", steering: ".kpi-steer", chat: "#panel-notes", log: "#panel-notes" };
  async function handleActions(actions, sim) {
    let sessionChanged = false;
    for (const a of actions) {
      if (a.type === "simulate") { setSim(a); if (sim) { app.sim = sim; renderSim(sim); } focusPanel("#panel-sim"); }
      else if ((a.type === "focus" || a.type === "open") && (PANEL[a.panel] || a.panel === "similar")) {
        if (a.panel === "similar") selectTab("tab-similar"); else if (a.panel === "pacing") selectTab("tab-pacing");
        focusPanel(PANEL[a.panel]);
      } else if (a.type === "close" && PANEL[a.panel]) {
        const el = $(PANEL[a.panel]); if (el) el.hidden = true;
      } else if (a.type === "runner") {
        sessionChanged = await openRunner(a.id, a.id === "demo" ? null : $("#holdout-check").checked);
        if (sessionChanged) {
          const confirmation = a.name ? `Opened ${a.name}.` : "Runner opened.";
          addReply(confirmation);
          if (hooks.speak) hooks.speak(confirmation);
        } else addReply("I couldn't open that runner. Please choose an existing record from the database.");
      }
      else if (a.type === "runner_database") openRunnerDatabase(a.query || "", { gender: a.gender || "", min_races: a.min_races || 0 });
      else if (a.type === "reveal") focusPanel("#kpi-acc");
    }
    return sessionChanged;
  }
  function focusPanel(sel) {
    const el = $(sel); if (!el) return;
    el.hidden = false;
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
  }

  async function setWeight(i, w) {
    try {
      const r = await api("PUT", `/sessions/${app.sid}/weights`, { index: i, weight: w });
      const it = r.card.items[0];
      app.changed = new Set([i]);
      render(r.state);
      if (it) { addSaid(`${it.race}: ${it.old.toFixed(2)} → ${it.new.toFixed(2)}`, "manual"); addCard(r.card); }
      adapt(r.adapt);
      runSim();
    } catch (e) { toast(e.message, "error"); render(app.st); }
  }

  async function undo() {
    if (!app.sid) return;
    try {
      const r = await api("POST", `/sessions/${app.sid}/undo`);
      render(r.state);
      addSaid("Undo", "undo");
      if (r.card.items.length) addCard(r.card); else addReply((r.card.notes || []).join(" ") || "Nothing to undo.");
      adapt(r.adapt);
      runSim();
    } catch (e) { toast(e.message, "error"); }
  }

  async function reveal() {
    try {
      const st = await api("POST", `/sessions/${app.sid}/reveal`);
      render(st);
      const ho = st.holdout;
      addReply(`Her real ${ho.label} result was ${fmt.time(ho.truth_s)}. The prediction is ${ho.error_pct.toFixed(2)}% off; `
        + `with equal weights it was ${ho.equal_weights_error_pct.toFixed(2)}% off.`);
      adapt(st.adapt);
    } catch (e) { toast(e.message, "error"); }
  }

  async function exportSession() {
    try {
      const data = await api("GET", `/sessions/${app.sid}/export`);
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
      const a = h("a", { href: url, download: `pacesense-session-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.json` });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      toast("Session exported. It can be replayed exactly with pacesense.steer.replay.");
    } catch (e) { toast(e.message, "error"); }
  }

  // ---------------------------------------------------------------- voice (continuous)
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const voice = { rec: null, on: false };
  function setupVoice() {
    if (!SR) return;
    const btn = $("#mic-btn"); btn.hidden = false;
    btn.addEventListener("click", () => (voice.on ? stopVoice() : startVoice()));
  }
  const FILLER_RE = /\b(um+|uh+|erm*|hmm+)\b/gi;
  const wordsOf = (t) => t.toLowerCase().replace(/[^a-z0-9' ]+/g, " ").split(/\s+/).filter(Boolean);
  /** PaceSense hearing its own voice through the speakers: most words are in what it is saying (or just said). */
  function isEcho(t) {
    const tts = hooks.tts();
    if (!tts.text || (!tts.speaking && performance.now() - tts.endedAt > 1500)) return false;
    const spoken = new Set(wordsOf(tts.text)), w = wordsOf(t);
    return w.length > 0 && w.filter((x) => spoken.has(x)).length / w.length >= 0.7;
  }
  function resetUtterance() { Object.assign(voice, { buf: [], conf: [], t0: 0, barge: false }); clearTimeout(voice.timer); }
  /** Send what was said once the coach has paused for the turn-taking silence (shorter for brisk coaches). */
  function flushUtterance() {
    const text = voice.buf.join(" ").replace(/\s+/g, " ").trim();
    if (!text) return resetUtterance();
    const now = performance.now(), dur = Math.max(300, now - voice.t0 - hooks.silenceMs()), n = wordsOf(text).length;
    const confs = voice.conf.filter((c) => c > 0), ended = hooks.tts().endedAt;
    const signals = {
      wpm: Math.round(n / (dur / 60000)), duration_ms: Math.round(dur), barge_in: voice.barge,
      fillers: (text.match(FILLER_RE) || []).length,
      confidence: confs.length ? +(confs.reduce((a, b) => a + b, 0) / confs.length).toFixed(2) : undefined,
      latency_ms: ended && voice.t0 > ended && voice.t0 - ended < 20000 ? Math.round(voice.t0 - ended) : undefined,
    };
    resetUtterance();
    $("#notes-input").value = "";
    hooks.transcript("", true, text);
    send(text, "voice", signals);
  }
  function startVoice() {
    const rec = new SR(); voice.rec = rec; voice.on = true; resetUtterance();
    rec.lang = "en-US"; rec.continuous = true; rec.interimResults = true;
    const input = $("#notes-input");
    rec.onresult = (e) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i], t = res[0].transcript;
        if (!t.trim() || isEcho(t)) continue;
        if (!voice.t0) voice.t0 = performance.now();
        if (hooks.tts().speaking && wordsOf(t).length >= 2) { voice.barge = true; hooks.bargeIn(); }   // talk over it
        if (res.isFinal) { voice.buf.push(t.trim()); voice.conf.push(res[0].confidence || 0); } else interim += t;
      }
      const shown = [...voice.buf, interim].join(" ").trim();
      if (shown) { input.value = shown; hooks.transcript(shown, false); }
      clearTimeout(voice.timer);
      if (voice.buf.length) voice.timer = setTimeout(flushUtterance, interim ? hooks.silenceMs() + 600 : hooks.silenceMs());
    };
    rec.onerror = (e) => { if (e.error === "not-allowed") { toast("Microphone permission was refused.", "error"); stopVoice(); } };
    rec.onend = () => { if (voice.on && !voice.paused) try { rec.start(); } catch { /* already started */ } };
    try { rec.start(); } catch { /* ignore */ }
    $("#mic-btn").setAttribute("aria-pressed", "true");
    hooks.voice(true);
  }
  function stopVoice(silent = false) {
    voice.on = false; if (voice.rec) voice.rec.stop();
    if (voice.buf && voice.buf.length) flushUtterance();
    $("#mic-btn").setAttribute("aria-pressed", "false");
    if (!silent) hooks.voice(false);
  }
  /** While PaceSense speaks, stop listening so it doesn't hear itself. */
  function pauseListening(p) {
    if (!voice.rec || !voice.on) return;
    if (p) { voice.paused = true; try { voice.rec.abort(); } catch { /* ignore */ } }
    else if (voice.paused) { voice.paused = false; try { voice.rec.start(); } catch { /* ignore */ } }
  }

  // ---------------------------------------------------------------- runner picker
  const dlg = $("#runner-dialog");
  function openRunnerDatabase(query = "", filters = {}) {
    dlg.showModal();
    $("#runner-search").value = query;
    $("#runner-gender-filter").value = filters.gender || "";
    $("#runner-min-races-filter").value = String(filters.min_races || 0);
    searchRunners(query);
    $("#runner-search").focus();
  }
  const searchRunners = debounce(async (q) => {
    const list = clear($("#runner-list"));
    $("#runner-db-count").textContent = "Searching the existing runner database…";
    try {
      const gender = $("#runner-gender-filter").value;
      const minRaces = $("#runner-min-races-filter").value || "0";
      const rows = await api("GET", `/runners?q=${encodeURIComponent(q)}&gender=${encodeURIComponent(gender)}&min_races=${encodeURIComponent(minRaces)}`);
      $("#runner-db-count").textContent = rows.length
        ? `Showing ${rows.length} matching existing runner records. Select one to open it; no new data is added.`
        : "No existing runners match that search.";
      if (!rows.length) list.append(h("li", { class: "muted", text: "No runners match. Try another name or runner ID." }));
      for (const r of rows) {
        const b = h("button", { type: "button", role: "option" },
          h("span", null, h("b", { text: r.runner_id === "demo" ? "Demo runner" : r.name }), r.runner_id === "demo" ? h("span", { class: "tag", text: "  recommended" }) : ""),
          h("span", { class: "best", text: r.best }),
          h("small", { text: [r.gender, r.age ? `${r.age} yrs` : "", `${r.n_races} races`, r.note || `best: ${r.best_race}`].filter(Boolean).join(" · ") }));
        b.addEventListener("click", () => {
          const cond = ($("#cond-choice input:checked") || {}).value || "ask";
          storage.set("pacesense.condition", cond);
          dlg.close(); openRunner(r.runner_id, r.runner_id === "demo" ? null : $("#holdout-check").checked, cond);
        });
        list.append(h("li", null, b));
      }
    } catch (e) { list.append(h("li", { class: "muted", text: e.message })); }
  }, 160);

  // ---------------------------------------------------------------- tabs
  function selectTab(id, byUser = false) {
    if (byUser) hooks.panel(id.replace("tab-", ""));
    $$(".tabs [role=tab]").forEach((t) => {
      const on = t.id === id;
      t.setAttribute("aria-selected", String(on)); t.tabIndex = on ? 0 : -1;
      $(`#${t.getAttribute("aria-controls")}`).hidden = !on;
    });
    storage.set("pacesense.tab", id);
    if (app.st) { if (id === "tab-similar") renderSimilar(app.st); if (id === "tab-pacing") renderPacing(app.st); }
  }

  // ---------------------------------------------------------------- theme
  function applyTheme(t) { document.documentElement.dataset.theme = t; storage.set("pacesense.theme", t); }

  // ---------------------------------------------------------------- adaptive bridge (see adaptive.js)
  function coachId() {
    let id = storage.get("pacesense.coach");
    if (!id) {
      const rnd = (crypto.getRandomValues ? [...crypto.getRandomValues(new Uint8Array(8))] : [Math.random() * 1e9])
        .map((x) => Math.floor(x).toString(16).padStart(2, "0")).join("");
      id = `coach-${rnd}`.slice(0, 40); storage.set("pacesense.coach", id);
    }
    return id;
  }
  const hooks = { transcript() {}, voice() {}, panel() {}, apply() {}, bargeIn() {}, silenceMs: () => 1000,
    tts: () => ({ speaking: false, text: "", endedAt: 0 }) };
  function adapt(a, opts = {}) { if (a) hooks.apply(a, opts); }
  window.PaceSense = {
    $, $$, h, s, clear, fmt, api, toast, app, storage, coachId, hooks,
    send, undo, reveal, selectTab, focusPanel, addReply, addSaid, addCard, scrollFeed, render, openRunner, runSim,
    setBusy, handleActions, applyTurn, animateSlider, setWeight,
    startVoice, stopVoice, pauseListening, sendReaction, hasMic: !!SR, voiceOn: () => voice.on,
    rerenderRaces: () => app.st && renderRaces(app.st),
    adapt,
  };

  // ---------------------------------------------------------------- wire up
  function init() {
    const saved = storage.get("pacesense.theme");
    applyTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
    $("#theme-btn").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

    $("#notes-form").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#notes-input").value; $("#notes-input").value = ""; send(t); });
    $("#notes-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#notes-form").requestSubmit(); }
    });
    $("#estimate-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const v = $("#estimate-input").value.trim();
      if (!/^\d:\d{2}(:\d{2})?$/.test(v)) { toast("Write the estimate as h:mm, for example 3:25.", "error"); return; }
      $("#estimate-input").value = "";
      send(`I think she can run ${v}`);
    });
    $("#undo-btn").addEventListener("click", undo);
    $("#export-btn").addEventListener("click", exportSession);
    $("#help-btn").addEventListener("click", () => $("#help-dialog").showModal());
    $("#runner-btn").addEventListener("click", () => openRunnerDatabase(""));
    $("#runner-search").addEventListener("input", (e) => searchRunners(e.target.value));
    $("#runner-gender-filter").addEventListener("change", () => searchRunners($("#runner-search").value));
    $("#runner-min-races-filter").addEventListener("change", () => searchRunners($("#runner-search").value));

    $$(".tabs [role=tab]").forEach((t) => {
      t.addEventListener("click", () => selectTab(t.id, true));
      t.addEventListener("keydown", (e) => {
        const tabs = $$(".tabs [role=tab]"), i = tabs.indexOf(t);
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          const n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]; selectTab(n.id, true); n.focus();
        }
      });
    });
    selectTab(storage.get("pacesense.tab") || "tab-pacing");

    $("#sim-form").addEventListener("input", () => { syncSimLabels(); runSim(); });
    $$("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      const [t, w, p] = b.dataset.preset.split(","); setSim({ temp_c: +t, wind_kmh: +w, profile: p }); runSim();
    }));
    syncSimLabels();

    document.addEventListener("keydown", (e) => {
      const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName) && document.activeElement.type !== "range";
      if (e.key === "/" && !typing) { e.preventDefault(); $("#notes-input").focus(); }
      else if (e.key === "?" && !typing) { $("#help-dialog").showModal(); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !typing) { e.preventDefault(); undo(); }
    });
    window.addEventListener("scroll", hideTip, { passive: true });
    let lastW = innerWidth;
    window.addEventListener("resize", debounce(() => {
      if (!app.st || innerWidth === lastW) return;
      lastW = innerWidth; renderPacing(app.st); renderSimilar(app.st); drawRangeBar(app.st); if (app.sim) renderSim(app.sim);
    }, 150));

    setupVoice();
    document.dispatchEvent(new CustomEvent("pacesense:ready"));
    api("GET", "/meta").then((m) => {
      app.meta = m;
      $("#foot-data").textContent = `Data: ${m.data.source} · ${m.data.runners.toLocaleString()} runners · ${m.data.cases.toLocaleString()} cases`;
    }).catch(() => {});
    resume();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
