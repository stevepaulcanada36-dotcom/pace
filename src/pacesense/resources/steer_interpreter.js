/* Browser port of src/pacesense/steer/intent.py (same lexicon, same rules). Needs a global LEX (the lexicon as JSON).
   Checked against the Python interpreter on every benchmark sentence by tests (see scripts/build_hud_demo.py). */
const Steer = (() => {
  const TEENS = {ten:10,eleven:11,twelve:12,thirteen:13,fourteen:14,fifteen:15,sixteen:16,seventeen:17,eighteen:18,nineteen:19};
  const POINT = {zero:0,one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9};
  const NUMW = {zero:0,nothing:0,none:0,half:.5,full:1,one:1,max:1,maximum:1,minimum:0};
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const cache = new Map();
  const re = p => { if (!cache.has(p)) cache.set(p, new RegExp("(?<![\\w'])" + esc(p).replace(/ /g, "\\s+") + "(?![\\w'])", "g")); const r = cache.get(p); r.lastIndex = 0; return r; };
  const cues = [];
  for (const [effect, g] of Object.entries(LEX.phrases)) for (const kind of ["verb","describe"]) for (const p of (g[kind] || [])) cues.push({phrase: String(p).toLowerCase(), effect, verb: kind === "verb"});
  cues.push({phrase:"less", effect:"down", verb:false}, {phrase:"more", effect:"up", verb:false});
  cues.sort((a, b) => b.phrase.length - a.phrase.length);
  const byPhrase = Object.fromEntries(cues.map(c => [c.phrase, c]));
  const soften = LEX.modifiers.soften, intensify = LEX.modifiers.intensify;

  function normalise(t){
    t = t.toLowerCase().replace(/’/g, "'").replace(/–/g, "-");
    t = t.replace(/\btwenty (ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\b/g, (m, a) => String(2000 + TEENS[a]));
    t = t.replace(/\btwenty twenty(?: (one|two|three|four|five|six|seven|eight|nine))?\b/g, (m, a) => String(2020 + (a ? POINT[a] : 0)));
    t = t.replace(/\btwo thousand (?:and )?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\b/g, (m, a) => String(2000 + (POINT[a] ?? TEENS[a])));
    t = t.replace(/\bpoint (zero|one|two|three|four|five|six|seven|eight|nine)\b/g, (m, a) => "0." + POINT[a]);
    t = t.replace(/\b'(\d\d)\b/g, (m, a) => "20" + a);
    return t.replace(/\s+/g, " ").trim();
  }
  const overlaps = (s, e, taken) => taken.some(([ts, te]) => s < te && ts < e);
  function findAll(text, phrases, taken){
    const out = [];
    for (const p of [...new Set(phrases)].sort((a, b) => b.length - a.length)){
      for (const m of text.matchAll(re(p))){ const s = m.index, e = s + m[0].length; if (overlaps(s, e, taken)) continue; taken.push([s, e]); out.push([s, e, p]); }
    }
    return out.sort((a, b) => a[0] - b[0]);
  }
  const wordsBefore = (text, pos, n) => (text.slice(0, pos).match(/[\w']+/g) || []).slice(-n);
  function hasBefore(text, pos, phrases, n){ const w = wordsBefore(text, pos, n).join(" "); for (const p of [...phrases].sort((a, b) => b.length - a.length)) if (re(p).test(w)) return p; return null; }
  const stripArticles = t => t.replace(/\b(?:the|her|his|their|a|an)\s+/g, "");
  function spanIn(text, phrase){ const pat = "(?<![\\w'])" + phrase.split(" ").map(esc).join("\\s+(?:(?:the|her|his|their|a|an)\\s+)?") + "(?![\\w'])"; const m = new RegExp(pat).exec(text); return m ? [m.index, m.index + m[0].length] : null; }
  const sentences = t => t.split(/(?<!\d)[.,](?!\d)|[;!?]+|\s+\bbut\b\s+|\s+\bhowever\b\s+|\s+\bwhereas\b\s+|\s+\balthough\b\s+/).map(s => (s || "").trim()).filter(Boolean);
  function order(races){ return races.map((r, i) => i).sort((a, b) => ((races[b].year || 0) - (races[a].year || 0)) || (b - a)); }
  function references(clause, races, taken, unknown){
    const refs = [], newest = order(races);
    const groups = {most_recent: newest.slice(0, 1), previous: newest.slice(1, 2), oldest: newest.slice(-1), recent_group: newest.slice(0, 2),
      older_group: newest.length <= 3 ? newest.slice(1) : newest.slice(2), all: races.map((r, i) => i)};
    const events = [...new Set(races.map(r => String(r.event || "").toLowerCase()).filter(Boolean))].sort((a, b) => b.length - a.length);
    for (const ev of events){
      for (const m of clause.matchAll(new RegExp("(?<![\\w'])" + esc(ev) + "(?:\\s+(?:in\\s+)?((?:19|20)\\d\\d))?(?![\\w'])", "g"))){
        const s = m.index, e = s + m[0].length; if (overlaps(s, e, taken)) continue;
        const year = m[1] ? +m[1] : null, idx = races.map((r, i) => i).filter(i => String(races[i].event).toLowerCase() === ev && (year === null || races[i].year === year));
        taken.push([s, e]); if (idx.length) refs.push({s, e, races: idx, pronoun: false}); else unknown.push(m[0]);
      }
    }
    for (const m of clause.matchAll(/(?<![\w.])((?:19|20)\d\d)(?![\w.])/g)){
      const s = m.index, e = s + m[0].length; if (overlaps(s, e, taken)) continue; taken.push([s, e]);
      const idx = races.map((r, i) => i).filter(i => races[i].year === +m[1]);
      if (idx.length) refs.push({s, e, races: idx, pronoun: false}); else unknown.push(m[1]);
    }
    for (const [key, phrases] of Object.entries(LEX.references)){ if (key === "pronoun") continue;
      for (const [s, e] of findAll(clause, phrases, taken)) if (groups[key] && groups[key].length) refs.push({s, e, races: groups[key], pronoun: false}); }
    for (const [s, e] of findAll(clause, LEX.references.pronoun || [], taken)) refs.push({s, e, races: [], pronoun: true});
    return refs.sort((a, b) => a.s - b.s);
  }
  const VALUE = "(0?\\.\\d+|1\\.0+|0|1|zero|nothing|none|half|full|one|max|maximum|minimum)";
  function explicitSets(clause, refs, taken){
    const out = [];
    if (!/\b(set|put|make|give|change|move|weight|weigh|slider|to|at)\b/.test(clause)) return out;
    const r = new RegExp("(?:\\b(?:to|at|of|=|weight|weighted)\\s+(?:a\\s+)?(?:weight\\s+(?:of\\s+)?)?)" + VALUE + "(?![\\w:])", "g");
    for (const m of clause.matchAll(r)){
      const vs = m.index + m[0].length - m[1].length, ve = m.index + m[0].length; if (overlaps(vs, ve, taken)) continue;
      let v = m[1] in NUMW ? NUMW[m[1]] : parseFloat(m[1]); if (!(v >= 0 && v <= 1)) continue;
      taken.push([m.index, ve]);
      const before = refs.filter(x => x.e <= m.index), target = before.length ? before[before.length - 1] : (refs[0] || null);
      out.push([target, Math.round(v * 10) / 10, m[0].trim()]);
    }
    return out;
  }
  const TIME = /(\d{1,2}:\d{2}(?::\d{2})?|\d\s*h(?:ours?|rs?)?\s*(?:and\s*)?\d{1,2}\s*(?:m(?:in(?:ute)?s?)?)?(?:\s*\d{1,2}\s*s(?:ec(?:ond)?s?)?)?|(?:sub|under)\s*\d(?::\d{2})?)/;
  function parseTime(raw){
    raw = raw.replace(/^(?:sub|under)\s*/, "");
    let m = raw.match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?$/); if (m) return +m[1] * 3600 + +m[2] * 60 + (+m[3] || 0);
    m = raw.match(/^(\d)\s*h(?:ours?|rs?)?\s*(?:and\s*)?(\d{1,2})?/); if (m) return +m[1] * 3600 + (+m[2] || 0) * 60;
    m = raw.match(/^(\d)$/); if (m) return +m[1] * 3600;
    return null;
  }
  function estimate(text){
    for (const cue of LEX.estimate_cues) for (const m of text.matchAll(re(cue))){
      const from = m.index + m[0].length, tail = text.slice(from, from + 40), tm = TIME.exec(tail); if (!tm) continue;
      const s = parseTime(tm[1]); if (s && s >= 7200 && s <= 25200) return [s, [from + tm.index, from + tm.index + tm[0].length]];
    }
    return [null, null];
  }
  function interpret(text, races, context){
    const norm = normalise(text), out = {text, actions: [], estimate: null, unresolved: [], unknown: [], context};
    if (!norm) return out;
    const bare = stripArticles(norm);
    for (const kind of ["undo", "reset"]){ const hit = findAll(bare, LEX.commands[kind] || [], []); if (hit.length){ out.actions.push({kind, races: [], value: null, cue: hit[0][2], conf: 1}); return out; } }
    const [est, estSpan] = estimate(norm); out.estimate = est;
    const rec = findAll(bare, LEX.commands.recency || [], []);
    const recSpans = rec.map(r => spanIn(norm, r[2])).filter(Boolean);
    if (rec.length) out.actions.push({kind: "recency", races: races.map((r, i) => i), value: null, cue: rec[0][2], conf: 1});
    let ctx = context;
    for (const clause of sentences(norm)){
      const offset = norm.indexOf(clause), taken = [];
      for (const sp of recSpans.concat(estSpan ? [estSpan] : [])) if (sp[0] >= offset && sp[1] <= offset + clause.length + 1) taken.push([sp[0] - offset, sp[1] - offset]);
      const unknown = [], refs = references(clause, races, taken, unknown); out.unknown.push(...unknown);
      const resolved = [];
      for (const r of refs){ if (r.pronoun){ const prior = resolved.filter(x => !x.pronoun); const idx = prior.length ? prior[prior.length - 1].races : (ctx != null ? [ctx] : []); if (idx.length) resolved.push({...r, races: idx}); } else resolved.push(r); }
      const named = resolved.filter(r => r.races.length);
      for (const [target, value, cue] of explicitSets(clause, named, taken)){
        const idx = target ? target.races : (ctx != null ? [ctx] : []);
        if (idx.length){ out.actions.push({kind: "set", races: idx, value, cue, conf: 1}); ctx = idx[idx.length - 1]; } else out.unresolved.push(cue);
      }
      const hits = findAll(clause, cues.map(c => c.phrase), taken);
      for (const [s, e, phrase] of hits){
        const cue = byPhrase[phrase], spec = LEX.effects[cue.effect];
        const neg = hasBefore(clause, s, LEX.negations, 3);
        let mod = 1; if (hasBefore(clause, s, soften.words, 3)) mod = soften.factor; else if (hasBefore(clause, s, intensify.words, 2)) mod = intensify.factor;
        const before = named.filter(r => r.e <= s), after = named.filter(r => r.s >= e);
        const target = cue.verb ? (after[0] || before[before.length - 1] || null) : (before[before.length - 1] || after[0] || null);
        let idx, conf = .8;
        if (!target){ if (ctx == null){ out.unresolved.push(phrase); continue; } idx = [ctx]; conf = .6; } else { idx = target.races; if (target.pronoun) conf = .6; }
        const label = neg ? neg + " " + phrase : phrase;
        if ("set" in spec && !neg) out.actions.push({kind: "set", races: idx, value: spec.set, cue: label, conf});
        else { let d = "shift" in spec ? spec.shift : (spec.set === 0 ? -.3 : .3); if (neg) d = -.5 * d; out.actions.push({kind: "shift", races: idx, value: Math.round(d * mod * 100) / 100, cue: label, conf}); }
        if (idx.length === 1) ctx = idx[0];
      }
      if (named.length && !hits.length){ const last = named[named.length - 1]; if (last.races.length === 1) ctx = last.races[0]; }
    }
    out.context = ctx; out.actions = merge(out.actions); return out;
  }
  function merge(actions){
    const singles = new Map(), keep = [];
    for (const a of actions){ if (a.kind === "set" || a.kind === "shift"){ for (const i of a.races){ if (!singles.has(i)) singles.set(i, []); singles.get(i).push(a); } } else keep.push(a); }
    for (const [i, acts] of singles){
      const sets = acts.filter(a => a.kind === "set");
      if (sets.length){ const a = sets[sets.length - 1]; keep.push({kind: "set", races: [i], value: a.value, cue: a.cue, conf: a.conf}); }
      else { const tot = Math.max(-1, Math.min(1, acts.reduce((x, a) => x + a.value, 0))); keep.push({kind: "shift", races: [i], value: Math.round(tot * 100) / 100, cue: [...new Set(acts.map(a => a.cue))].join(" + "), conf: Math.min(...acts.map(a => a.conf))}); }
    }
    const ord = {undo: 0, reset: 1, recency: 2, set: 3, shift: 3};
    return keep.sort((a, b) => (ord[a.kind] - ord[b.kind]) || ((a.races[0] ?? -1) - (b.races[0] ?? -1)));
  }
  function round1(x){ // Python round(x, 1): half to even on the decimal representation
    const y = x * 10, f = Math.floor(y), d = y - f;
    if (Math.abs(d - .5) < 1e-9) return (f % 2 === 0 ? f : f + 1) / 10;
    return Math.round(y) / 10;
  }
  function recencyWeights(years, h, floor){ const latest = Math.max(...years); return years.map(y => round1(Math.min(1, Math.max(floor, Math.pow(2, -(latest - y) / h))))); }
  return {interpret, recencyWeights, round1};
})();
