/* PaceSense command center, running entirely in the browser (no server).
   A port of src/pacesense/hud/session.py, steer/session.py, steer/evidence.py, hud/commands.py and hud/sim.py
   for one embedded demo runner. Needs globals: LEX (lexicon), DEMO (runner data exported from Python), Steer
   (steer_interpreter.js). scripts/build_hud_demo.py builds the page and checks its numbers against Python. */
window.PACESENSE_LOCAL = (() => {
"use strict";
const KM = [5, 5, 5, 5, 5, 5, 5, 5, 2.195], MARATHON = 42.195, LABELS = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-42.2"];
const fmt = s => { s = Math.round(s); return `${Math.floor(s / 3600)}:${String(Math.floor(s % 3600 / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`; };
const dur = s => { const r = Math.round(Math.abs(s)); return r >= 3600 ? fmt(r) : `${Math.floor(r / 60)}:${String(r % 60).padStart(2, "0")}`; };
const signed = s => { const r = Math.round(Math.abs(s)); return `${s > 0 ? "+" : "−"}${Math.floor(r / 60)}:${String(r % 60).padStart(2, "0")}`; };
const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const R = DEMO.races, N = R.length;

/* ---------- model (MarathonModel.combine) ---------- */
function combine(w){
  const sw = w.reduce((a, b) => a + b, 0);
  const finish = R.reduce((a, r, i) => a + w[i] * r.pred_s, 0) / sw;
  let pw = 0; const prof = new Array(9).fill(0);
  R.forEach((r, i) => { if (r.profile && w[i] > 0){ pw += w[i]; r.profile.forEach((v, k) => prof[k] += w[i] * v); } });
  const profile = pw > 0 ? prof.map(v => v / pw) : new Array(9).fill(1);
  const vals = [], wts = [];
  R.forEach((r, i) => { if (w[i] > 0) r.cases.forEach((v, k) => { vals.push(v); wts.push(r.cw[k] * w[i] / sw); }); });
  const ord = vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]), v = ord.map(i => vals[i]), ww = ord.map(i => wts[i]);
  let c = 0; const cum = ww.map(x => (c += x) - .5 * x), tot = ww.reduce((a, b) => a + b, 0);
  const q = p => { const t = p * tot; if (t <= cum[0]) return v[0];
    for (let i = 1; i < cum.length; i++) if (t <= cum[i]) return v[i - 1] + (v[i] - v[i - 1]) * (t - cum[i - 1]) / (cum[i] - cum[i - 1]);
    return v[v.length - 1]; };
  return {finish, low: q(.2), high: q(.8), paces: profile.map(p => p * finish / MARATHON), n: R.reduce((a, r, i) => a + (w[i] > 0 ? r.n : 0), 0)};
}

/* ---------- evidence (steer/evidence.py) ---------- */
const EV = R.map((r, i) => { const o = R.filter((_, j) => j !== i).map(x => x.finish_s), med = median(o),
  mad = median(o.map(x => Math.abs(x - med))) * 1.4826, z = (r.finish_s - med) / Math.max(mad, .02 * med);
  return {z, med, direction: Math.abs(z) < 1 ? 0 : (z > 0 ? 1 : -1),
          label: Math.abs(z) < 1 ? "typical for this runner" : z > 0 ? "slower than this runner's other races" : "faster than this runner's other races"}; });
const mins = s => { const m = Math.abs(s) / 60; return m >= 1.5 ? `${Math.round(m)} min` : `${Math.round(Math.abs(s))} s`; };
function checkChange(i, old, nw){
  const name = raceName(i), ev = EV[i];
  if (Steer.round1(nw - old) === 0) return ["no_change", "No change."];
  const lowering = nw < old, gap = mins(R[i].finish_s - ev.med);
  if (ev.direction === 0) return ["your_knowledge", `The times do not single out ${name} (within the usual spread of this runner's races), so this change rests on what you know about it.`];
  const slow = ev.direction > 0;
  if (lowering === slow) return ["supported", `The data agrees: ${name} was ${gap} ${slow ? "slower" : "faster"} than this runner's other races.`];
  return ["against_data", lowering ? `${name} was one of this runner's faster races (${gap} faster than the others); counting it less makes the target slower.`
                                   : `${name} was one of this runner's slower races (${gap} slower than the others); counting it more makes the target slower.`];
}
function influence(w){
  const sw = w.reduce((a, b) => a + b, 0), f = R.reduce((a, r, i) => a + w[i] * r.pred_s, 0) / sw;
  return R.map((r, i) => ({pull_s: w[i] * (r.pred_s - f) / sw}));
}
function steering(initial, now, est){
  if (est == null) return {index: null, coach_estimate_s: null, label: "Say your own estimate (\"I think she can run 3:25\") to see whose view the model is moving towards."};
  const gap = est - initial; if (Math.abs(gap) < 60) return {index: null, coach_estimate_s: est, label: "Your estimate and the model's agreed from the start."};
  const a = (now - initial) / gap;
  const label = Math.abs(a) < .15 ? "The model is where the data put it." : a < 0 ? "The model has moved away from your estimate."
    : a < .85 ? `The model has moved ${Math.round(a * 100)}% of the way towards your estimate.`
    : a <= 1.15 ? "The model now matches your own estimate. Is that the data, or the sliders agreeing with you?" : "The model has been pushed beyond your own estimate.";
  return {index: Math.round(a * 1000) / 1000, coach_estimate_s: est, label};
}

/* ---------- the session (steer/session.py + hud/session.py) ---------- */
const S = {w: new Array(N).fill(.5), ctx: null, est: null, turns: [], undo: [], log: [], moves: 0, reasoned: 0, revealed: false, reactions: []};
const initial = combine(new Array(N).fill(.5));
const raceName = i => `${R[i].event} ${R[i].year}`;
function setW(turn, i, nw, cue, reasoned, sayUnchanged){
  nw = Steer.round1(Math.min(1, Math.max(0, nw))); const old = S.w[i];
  if (nw === old){ if (sayUnchanged){ const what = old === 0 ? "counts for nothing" : old === 1 ? "counts fully" : `is at ${old.toFixed(1)}`;
    turn.notes.push(`${raceName(i)} already ${what} (heard "${cue}").`); } return; }
  const trial = S.w.slice(); trial[i] = nw;
  if (trial.every(x => x <= 0)){ turn.notes.push(`${raceName(i)} left at ${old.toFixed(1)}: give at least one race with a prediction a weight above 0.`); return; }
  const before = combine(S.w).finish; S.w[i] = nw; const after = combine(S.w).finish;
  const [verdict, evidence] = cue === "undo" ? ["undo", "Back to the value before the last change."] : checkChange(i, old, nw);
  turn.applied.push({index: i, race: raceName(i), old, new: nw, cue, reasoned, verdict, evidence});
  if (cue !== "undo"){ S.moves++; if (reasoned) S.reasoned++; }
  const d = after - before, eff = Math.round(d) ? `Model ${dur(d)} ${d < 0 ? "faster" : "slower"} (${fmt(after)}), your plan ${dur(d)} ${d < 0 ? "faster" : "slower"} (${fmt(after)}).` : "No change to the finish times.";
  const said = cue === "slider" ? "Set by hand." : cue === "undo" ? "Undone." : `You said: "${cue}".`;
  S.log.unshift(`${raceName(i)} weight changed from ${old.toFixed(1)} to ${nw.toFixed(1)}. ${eff} ${said}`);
}
function begin(text, source){ S.undo.push(S.w.slice()); return {n: S.turns.length + 1, text, said: text, source, applied: [], notes: [], finish_before_s: combine(S.w).finish, estimate_s: null, interpreter: "rules"}; }
function end(turn){ turn.finish_after_s = combine(S.w).finish; turn.delta_s = turn.finish_after_s - turn.finish_before_s; if (!turn.applied.length) S.undo.pop(); S.turns.push(turn); return turn; }
function undo(text){
  const turn = {n: S.turns.length + 1, text, said: text, source: "undo", applied: [], notes: [], finish_before_s: combine(S.w).finish, estimate_s: null};
  if (!S.undo.length) turn.notes.push("Nothing to undo."); else S.undo.pop().forEach((w, i) => setW(turn, i, w, "undo", true));
  turn.finish_after_s = combine(S.w).finish; turn.delta_s = turn.finish_after_s - turn.finish_before_s; S.turns.push(turn); return turn;
}
function hear(text, source){
  const it = Steer.interpret(text, R, S.ctx); S.ctx = it.context;
  if (it.actions.some(a => a.kind === "undo")) return undo(text);
  const turn = begin(text, source);
  if (it.estimate != null){ S.est = turn.estimate_s = it.estimate; turn.notes.push(`Your estimate: ${fmt(it.estimate)}.`); }
  for (const a of it.actions){
    if (a.kind === "reset") R.forEach((_, i) => setW(turn, i, .5, a.cue, true));
    else if (a.kind === "recency"){ Steer.recencyWeights(R.map(r => r.year), LEX.recency.half_life_years, LEX.recency.floor)
      .forEach((w, i) => { if (S.w[i] < .5 && w > S.w[i]) return; setW(turn, i, w, a.cue, true); }); }
    else if (a.kind === "set") a.races.forEach(i => setW(turn, i, a.value, a.cue, !/^(to|at|of|=|weight|weighted)\b/.test(a.cue), true));
    else if (a.kind === "shift") a.races.forEach(i => setW(turn, i, S.w[i] + a.value, a.cue, true, true));
  }
  it.unknown.forEach(u => turn.notes.push(`This runner has no race '${u}' among these races.`));
  it.unresolved.forEach(u => turn.notes.push(`Heard "${u}" but not which race it is about. Name the race or the year.`));
  if (!it.actions.length && it.estimate == null && !it.unknown.length && !it.unresolved.length) turn.notes.push("Heard you, but nothing in that changes a race weight.");
  return end(turn);
}
function manual(i, w){ const turn = begin(`${raceName(i)} slider`, "manual"); setW(turn, i, w, "slider", false); return end(turn); }
const errPct = f => 100 * Math.abs(f - DEMO.truth_s) / DEMO.truth_s;

function state(){
  const m = combine(S.w), inf = influence(S.w);
  const verdict = m.finish < m.low ? "ambitious" : m.finish > m.high ? "cautious" : "realistic";
  return {
    runner: {id: "demo", name: DEMO.runner.name}, mode: "demo",
    races: R.map((r, i) => ({index: i, event: r.event, year: r.year, finish_s: r.finish_s, finish: r.finish, weight: S.w[i], prediction_s: r.pred_s,
                             evidence: {z: EV[i].z, label: EV[i].label, direction: EV[i].direction}, influence: inf[i]})),
    prediction: {finish_s: m.finish, finish: fmt(m.finish), low_s: m.low, high_s: m.high, initial_s: initial.finish, initial: fmt(initial.finish), n_cases: m.n},
    steering: steering(initial.finish, m.finish, S.est),
    reasoned_share: S.moves ? S.reasoned / S.moves : null,
    change_log: S.log.slice(0, 20),
    plan: {paces: m.paces, labels: LABELS, finish_s: m.paces.reduce((a, p, k) => a + p * KM[k], 0), check: {verdict}},
    similar: R.map((r, i) => ({race_index: i, n: r.n, finishes_s: [...r.cases].sort((a, b) => a - b), median_s: median(r.cases)})),
    holdout: {label: DEMO.truth_label, revealed: S.revealed, truth_s: S.revealed ? DEMO.truth_s : null,
              error_pct: S.revealed ? errPct(m.finish) : null, equal_weights_error_pct: S.revealed ? errPct(initial.finish) : null},
    reactions: reactions(), ai: false, local: true,
  };
}
const reactions = () => ({n: S.reactions.length, agree: S.reactions.filter(r => r.value === "agree").length,
  unsure: S.reactions.filter(r => r.value === "unsure").length, disagree: S.reactions.filter(r => r.value === "disagree").length});

/* ---------- feedback loop (hud/session.py) ---------- */
const VW = {supported: "the data agrees", your_knowledge: "this rests on your knowledge", against_data: "this goes against the data", undo: "undone", no_change: "no change"};
function card(turn){
  const st = state(), pr = st.prediction, items = turn.applied.map(a => ({race: a.race, old: a.old, new: a.new, cue: a.cue, verdict: a.verdict, verdict_text: VW[a.verdict], evidence: a.evidence}));
  const c = {n: turn.n, said: turn.said || turn.text, items, notes: turn.notes, target: pr.finish, target_delta: items.length ? signed(turn.finish_after_s - turn.finish_before_s) : null,
             range: [fmt(pr.low_s), fmt(pr.high_s)], steering: st.steering, suggestions: []};
  if (items.length){ const b = errPct(turn.finish_before_s), a = errPct(turn.finish_after_s);
    c.accuracy = {closer: a < b - 1e-9, same: Math.abs(a - b) < 1e-9, before_pct: b, after_pct: a, equal_weights_pct: errPct(initial.finish),
                  text: a < b - 1e-9 ? "closer to her real result" : a > b + 1e-9 ? "further from her real result" : "no closer to her real result"}; }
  const v = new Set(items.map(i => i.verdict));
  if (v.has("against_data")) c.suggestions.push("undo", "Why does the data disagree?");
  if (v.has("your_knowledge")) c.suggestions.push("show similar runners", "Explain the likely range");
  if (v.has("supported")) c.suggestions.push("simulate a hot day");
  if (st.steering.index != null && st.steering.index > .85) c.suggestions.unshift("Is that the data, or me?");
  if (!items.length && !turn.notes.length) c.suggestions.push("2010 was a hot day", "trust the recent races more");
  c.suggestions = [...new Set(c.suggestions)].slice(0, 4);
  return c;
}
function cardText(c){
  if (!c.items.length) return c.notes.join(" ") || "Nothing changed.";
  const parts = c.items.map(i => `${i.race} ${i.old.toFixed(1)} → ${i.new.toFixed(1)}: ${i.verdict_text}.`);
  parts.push(`Target ${c.target} (${c.target_delta}).`);
  if (c.accuracy) parts.push(S.revealed ? `That is ${c.accuracy.text} (${c.accuracy.after_pct.toFixed(2)}% off).` : `That moved the prediction ${c.accuracy.text.split(" to ")[0]}.`);
  return parts.join(" ");
}

/* ---------- one command bar (hud/commands.py) ---------- */
const PANELS = {races: ["races", "sliders", "weights", "past races"], prediction: ["prediction", "target", "forecast", "likely range", "range"],
  pacing: ["pacing", "pace plan", "splits", "pace band"], similar: ["similar runners", "neighbours", "neighbors", "similar", "lookalikes"],
  simulator: ["simulator", "simulation", "race day", "race-day", "course"], feedback: ["feedback", "evidence", "checks", "data checks"],
  steering: ["steering", "steering meter", "whose view", "meter"], chat: ["chat", "conversation", "assistant"], log: ["change log", "log", "history of changes"],
  record: ["track record", "results", "outcomes"]};
const alts = ws => [...ws].sort((a, b) => b.length - a.length).map(w => w.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&")).join("|");
const PANEL_RE = new RegExp(`\\b(show|open|bring up|display|hide|close|remove|minimi[sz]e|focus on|zoom (?:in )?on|pin)\\s+(?:me\\s+)?(?:the\\s+|my\\s+|her\\s+|his\\s+)?(${alts(Object.values(PANELS).flat())})(?:\\s+panel)?\\b`, "gi");
const PW = ["coach", "runner", "athlete", "researcher", "research", "study"];
const PERSP_RE = new RegExp(`\\b(?:switch|change|go)\\s+(?:to\\s+)?(?:the\\s+)?(${alts(PW)})\\s+(?:view|mode|perspective)\\b|\\b(${alts(PW)})\\s+(?:view|mode|perspective)\\b`, "i");
const SIM_RE = /\b(simulate|simulation|run the race|play the race|replay the race|race[- ]day|what if)\b/i;
const TEMP_RE = /(-?\d{1,2})\s*(?:°\s*c|degrees(?:\s+celsius)?|celsius|c\b)/i;
const WIND_RE = /(\d{1,2})\s*(?:km\/?h|kph|kilometres per hour|kilometers per hour)\s*(?:head)?wind|(?:head)?wind\s*(?:of\s*)?(\d{1,2})\s*(?:km\/?h|kph)?/i;
const CALM_RE = /\b(calm mode|focus mode|quiet mode|stop (?:rotating|spinning|moving)|pause the hologram|less motion|reduce motion)\b/i;
const SPIN_RE = /\b(start (?:rotating|spinning)|rotate (?:her|him|the (?:body|hologram|runner))|resume)\b/i;
const LAYOUT_RE = /\b(reset (?:the )?layout|tidy (?:up )?(?:the )?(?:screen|panels)|arrange (?:the )?panels)\b/i;
const REVEAL_RE = /\b(reveal|show me the (?:real|actual|true) (?:best|result)|how (?:close|accurate) (?:am i|was i|is it))\b/i;
const DEMO_RE = /\b(demo runner|new demo|another runner|next runner)\b/i;
function uiCommands(text){
  const actions = [], said = []; let rest = text;
  for (const m of text.matchAll(PANEL_RE)){ const verb = m[1].toLowerCase(), word = m[2].toLowerCase(), panel = Object.keys(PANELS).find(k => PANELS[k].includes(word));
    const kind = ["hide", "close", "remove", "minimise", "minimize"].includes(verb) ? "close" : "focus"; actions.push({type: kind, panel});
    said.push(`${kind === "close" ? "hid" : "opened"} ${panel}`); rest = rest.replace(m[0], " "); }
  const pm = text.match(PERSP_RE);
  if (pm){ const w = (pm[1] || pm[2]).toLowerCase(), p = ["coach"].includes(w) ? "coach" : ["runner", "athlete"].includes(w) ? "runner" : "researcher";
    actions.push({type: "perspective", value: p}); said.push(`switched to the ${p} view`); rest = rest.replace(pm[0], " "); }
  if (SIM_RE.test(text)){ const t = text.match(TEMP_RE), w = text.match(WIND_RE);
    const temp = t ? +t[1] : /\b(hot|heat|warm|humid)\b/i.test(text) ? 27 : /\b(cold|freezing)\b/i.test(text) ? 5 : 12;
    const wind = w ? +(w[1] || w[2]) : /\bwindy|strong wind/i.test(text) ? 25 : 0;
    const prof = /fast start|goes out fast|starts too fast|start fast|too fast/i.test(text) ? "fast_start" : /fade|fades|late fade|hits the wall|hit the wall|blows up/i.test(text) ? "late_fade" : "on_plan";
    actions.push({type: "simulate", temp_c: temp, wind_kmh: wind, profile: prof});
    said.push(`simulated the race at ${temp} °C` + (wind ? `, ${wind} km/h headwind` : "") + (prof === "on_plan" ? "" : `, ${prof.replace("_", " ")}`)); rest = ""; }
  if (CALM_RE.test(text)){ actions.push({type: "calm", value: true}); said.push("calm mode on"); rest = rest.replace(CALM_RE, " "); }
  else if (SPIN_RE.test(text)){ actions.push({type: "calm", value: false}); said.push("motion back on"); rest = rest.replace(SPIN_RE, " "); }
  if (DEMO_RE.test(text)){ actions.push({type: "runner", id: "demo"}); said.push("restarted the demo"); rest = rest.replace(DEMO_RE, " "); }
  if (LAYOUT_RE.test(text)){ actions.push({type: "layout", value: "reset"}); said.push("tidied the panels"); rest = rest.replace(LAYOUT_RE, " "); }
  if (REVEAL_RE.test(text)){ actions.push({type: "reveal"}); said.push("revealed the hidden best race"); rest = rest.replace(REVEAL_RE, " "); }
  rest = rest.replace(/\b(and|then|also|please|can you|could you)\b/gi, " ").replace(/\s+/g, " ").trim();
  return {actions, said, rest};
}
function ask(q){
  const t = q.toLowerCase(), st = state(), pr = st.prediction;
  if (/why.*disagree|against the data/.test(t)) return "The check compares each race with her other races. Lowering one of her faster races, or raising a slower one, moves the target the opposite way to what the times suggest. It may still be right if you know something the times cannot show; if not, say “undo”.";
  if (/data,? or me|is that the data/.test(t)) return `The steering index compares how far the model has moved with the gap between the model and your own estimate. ${st.steering.label}`;
  if (/range|wide|uncertain|sure/.test(t)) return `The likely range is ${fmt(pr.low_s)} to ${fmt(pr.high_s)}: the 20th to 80th percentile of what ${pr.n_cases} similar runners went on to run, weighted by your race weights. About 8 in 10 runners like her finished inside it. It narrows when the races you trust point to similar runners.`;
  if (/pull|which race|influen|matter/.test(t)){ const i = st.races.reduce((b, r, k) => Math.abs(r.influence.pull_s) > Math.abs(st.races[b].influence.pull_s) ? k : b, 0), r = st.races[i];
    return `${r.event} ${r.year} pulls the prediction most: ${dur(Math.abs(r.influence.pull_s))} ${r.influence.pull_s > 0 ? "slower" : "faster"} at weight ${r.weight.toFixed(1)}. Its own prediction is ${fmt(r.prediction_s)}.`; }
  if (/realistic|my plan|ambitious|cautious/.test(t)) return `The plan finishes in ${fmt(st.plan.finish_s)}, inside the likely range (${fmt(pr.low_s)}–${fmt(pr.high_s)}), so it is ${st.plan.check.verdict}.`;
  if (/similar|like her|neighbo/.test(t)) return `For each race, the model finds the ${R[0].n} runners whose earlier race looked most like hers and reuses how much they improved at their best. Their implied times for her are the dots in “Runners like her”.`;
  if (/predict|target|time|how fast/.test(t)) return `The model's target is ${pr.finish} (likely ${fmt(pr.low_s)}–${fmt(pr.high_s)}). With equal weights it was ${pr.initial}.`;
  return "This demo runs in your browser without the AI assistant, so I answer a few questions: the likely range, which race pulls the prediction most, whether the plan is realistic, similar runners. Talk about her races to move the sliders, or ask for the simulator.";
}

/* ---------- simulator (hud/sim.py + racewatch) ---------- */
function paceAt(p, d){ let acc = 0; for (let k = 0; k < 9; k++){ acc += KM[k]; if (d < acc || k === 8) return p[k]; } return p[8]; }
function targetElapsed(p, d){ let t = 0, acc = 0; for (let k = 0; k < 9; k++){ const seg = Math.min(KM[k], Math.max(0, d - acc)); t += seg * p[k]; acc += KM[k]; } return t; }
function factor(d, temp, wind, plan){ const heat = Math.max(0, temp - 11) * .30 / 100 * (.5 + d / MARATHON) * Math.sqrt(Math.max(plan, 7200) / 12600);
  const v = Math.max(0, wind) / 3.6; return 1 + heat + .05 / 100 * v * v * .5; }
function status(p, d, el){
  const plan = p.reduce((a, x, k) => a + x * KM[k], 0), target = targetElapsed(p, d), delta = el - target, remKm = MARATHON - d, proj = el + plan - target;
  const req = remKm > 1e-9 ? (plan - el) / remKm : NaN, tp = paceAt(p, d), pace = s => `${Math.floor(Math.round(s) / 60)}:${String(Math.round(s) % 60).padStart(2, "0")}/km`;
  if (d >= MARATHON - 1e-9) return ["done", `Finished in ${fmt(el)}. Plan was ${fmt(plan)} (${Math.round(delta) ? `${dur(delta)} ${delta < 0 ? "faster" : "slower"}` : "on the dot"}).`];
  if (d < .05 && el < 30) return ["start", `Start at ${pace(tp)}. Target finish ${fmt(plan)}.`];
  if (Math.abs(delta) <= DEMO.on_plan_s) return ["on", `On plan. Hold ${pace(tp)}.`];
  if (delta < 0) return ["ahead", d < 30 ? `${dur(delta)} ahead of plan. Settle back to ${pace(tp)}.` : `${dur(delta)} ahead with ${remKm.toFixed(1)} km to go. Hold ${pace(tp)}.`];
  if (remKm > 1e-9 && req < tp * (1 - DEMO.slipping)) return ["slipping", `${dur(delta)} behind. Plan pace from here finishes in about ${fmt(proj)}; ${pace(req)} needed for ${fmt(plan)}.`];
  return ["behind", `${dur(delta)} behind plan. ${pace(req)} from here finishes on time.`];
}
function simulate(temp, wind, profile){
  if (!(temp >= -10 && temp <= 40)) throw new Error("temperature must be between -10 and 40 °C");
  if (!(wind >= 0 && wind <= 60)) throw new Error("headwind must be between 0 and 60 km/h");
  const m = combine(S.w), p = m.paces, plan = p.reduce((a, x, k) => a + x * KM[k], 0);
  const ds = []; for (let i = 0; i * .1 < MARATHON - 1e-9; i++) ds.push(Math.round(i * .1 * 1000) / 1000); ds.push(MARATHON);
  const el = [0]; let t = 0;
  for (let i = 0; i < ds.length - 1; i++){ const a = ds[i], b = ds[i + 1]; let f = 1 + DEMO.noise[Math.floor(a)];
    if (profile === "fast_start") f *= a < 10 ? .97 : a >= 30 ? 1.05 : 1; else if (profile === "late_fade") f *= a >= 30 ? 1.06 : 1;
    t += paceAt(p, (a + b) / 2) * f * (b - a); el.push(t); }
  const sim = [0]; for (let i = 0; i < ds.length - 1; i++) sim.push(sim[i] + (el[i + 1] - el[i]) * factor((ds[i] + ds[i + 1]) / 2, temp, wind, plan));
  const interp = d => { for (let i = 1; i < ds.length; i++) if (d <= ds[i]) return sim[i - 1] + (sim[i] - sim[i - 1]) * (d - ds[i - 1]) / (ds[i] - ds[i - 1]); return sim[sim.length - 1]; };
  const lo = m.low / plan, hi = m.high / plan, frames = [];
  const kms = []; for (let k = 0; k < MARATHON - 1e-9; k += .5) kms.push(k); kms.push(MARATHON);
  for (const km of kms){ const s = interp(km), pl = targetElapsed(p, km), [level, cue] = status(p, km, s);
    let j = ds.findIndex(d => d > km) - 1; if (j < 0 || j >= ds.length - 1) j = ds.length - 2;
    const pace = (sim[j + 1] - sim[j]) / Math.max(ds[j + 1] - ds[j], 1e-9);
    frames.push({km: Math.round(km * 1000) / 1000, plan_s: pl, sim_s: s, low_s: pl * lo, high_s: pl * hi, delta_s: s - pl, pace_s: pace, level, cue}); }
  const fin = frames[frames.length - 1].sim_s, fb = frames.find(f => f.delta_s > 60);
  return {frames, summary: {plan_finish_s: plan, sim_finish_s: fin, delta_s: fin - plan, low_s: m.low, high_s: m.high, inside_range: m.low <= fin && fin <= m.high,
    first_minute_behind_km: fb ? fb.km : null, profile, temp_c: temp, wind_kmh: wind,
    conditions_pct: Math.round(100 * (factor(21.1, temp, wind, plan) - 1) * 100) / 100}, assumptions: "Heat and wind effects are stated assumptions, not fitted on data."};
}

/* ---------- routing (web/hud_api.py) ---------- */
function command(text){
  const {actions, said, rest} = uiCommands(text), used = actions.length ? ["ui"] : [];
  let reply = "", turn = null, sim = null, steered = false;
  if (rest && rest.length > 2){
    const it = Steer.interpret(rest, R, S.ctx), strong = it.actions.filter(a => ["reset", "undo", "recency"].includes(a.kind) || a.conf >= .8);
    if (strong.length || it.estimate != null || (it.actions.length && !actions.length)){
      const t = hear(rest, "typed");
      if (t.applied.length || t.estimate_s != null || t.source === "undo" || t.notes.length){ turn = {...t, said: text}; actions.push({type: "refresh"}); used.push("steer"); steered = true; }
    }
  }
  const c = turn ? card(turn) : null;
  if (steered) reply = cardText(c);
  else if (rest && rest.length > 2 && !(actions.length && rest.split(" ").length < 3)){ reply = ask(rest); used.push("assistant"); }
  if (said.length) reply = (`Done: ${said.join(", ")}.` + (reply ? " " + reply : "")).trim();
  if (!reply) reply = "I heard you, but nothing in that changes a race or the screen. Try “2010 was a hot day”, “simulate a hot day”, “switch to runner view” or ask a question.";
  for (const a of actions){ if (a.type === "reveal") S.revealed = true; if (a.type === "simulate") sim = simulate(a.temp_c, a.wind_kmh, a.profile);
    if (a.type === "runner") reset(); }
  return {reply, actions, turn, card: c, used, steps: [], simulation: sim, state: state()};
}
function reset(){ Object.assign(S, {w: new Array(N).fill(.5), ctx: null, est: null, turns: [], undo: [], log: [], moves: 0, reasoned: 0, revealed: false, reactions: []}); }

return async function local(method, path, body = {}){
  await new Promise(r => setTimeout(r, 30));
  if (path === "/api/hud/sessions" && method === "POST"){ reset(); return {id: "local-demo-session", state: state()}; }
  if (path === "/api/hud/track-record") return {n: 0};
  const m = path.match(/^\/api\/hud\/sessions\/[^/]+(\/\w+)?$/); if (!m) throw new Error("not available in the browser demo");
  const what = m[1] || "";
  if (!what) return state();
  if (what === "/command"){ if (!body.text || !String(body.text).trim()) throw new Error("text is required"); return command(String(body.text)); }
  if (what === "/weights"){ const t = manual(body.index, body.weight); return {turn: t, card: card(t), state: state()}; }
  if (what === "/simulate") return simulate(+body.temp_c, +body.wind_kmh, body.profile || "on_plan");
  if (what === "/reveal"){ S.revealed = true; return state(); }
  if (what === "/react"){ if (!["agree", "unsure", "disagree"].includes(body.value)) throw new Error("value must be agree, unsure or disagree");
    S.reactions.push({turn: body.turn, value: body.value}); return reactions(); }
  throw new Error("not available in the browser demo");
};
})();
