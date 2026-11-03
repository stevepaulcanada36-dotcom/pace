/* PaceSense Studio — the adaptive layer in the browser.
 *
 * The server's feedback loops decide; this file shows the result: it re-arranges the layout, opens the voice
 * stage, renders PaceSense's questions with tap-or-say choices, keeps the loops bar and the coach profile up
 * to date, and speaks short replies. It plugs into studio.js through window.PaceSense.hooks. */
"use strict";
(() => {
  document.addEventListener("pacesense:ready", install, { once: true });

  function install() {
    const P = window.PaceSense;
    const { $, $$, h, s, clear, api, toast, app, storage } = P;
    const A = { adapt: null, promptT: null, layout: null, voiceMode: false, speakOn: null, loopNames: {} };
    const LAYOUT_NAMES = { standard: "Standard", voice: "Voice stage", analysis: "Analysis first", races: "Races first",
      summary: "Summary first", guided: "Guided" };
    const TIMESCALE = { instant: "now", session: "this session", "cross-session": "across sessions", population: "all coaches" };

    // ------------------------------------------------------------ apply what the loops decided
    function apply(a, opts = {}) {
      A.adapt = a;
      (a.loops || []).forEach((l) => { A.loopNames[l.key] = l.name; });
      renderPersonaChip(a.persona);
      renderLoops(a);
      const ui = a.ui || {};
      document.body.dataset.explain = ui.explain || "normal";
      if (ui.layout !== A.layout) {
        const first = A.layout === null;
        A.layout = ui.layout;
        document.body.dataset.layout = ui.layout;
        if (!first && ui.analysis_tab) P.selectTab(`tab-${ui.analysis_tab}`);
        if (first && ui.analysis_tab && ui.layout === "analysis") P.selectTab(`tab-${ui.analysis_tab}`);
      }
      setStage(ui.layout === "voice");
      app.spotlight = ui.spotlight || [];
      P.rerenderRaces();
      renderStage();
      if (A.speakOn === null) A.speakOn = readSpeakPref(ui);
      $("#speak-btn").setAttribute("aria-pressed", String(!!A.speakOn));

      for (const ad of a.adaptations || []) {
        if (!ad.message) continue;
        if (ad.kind === "insight" || ad.kind === "persona") sysNote(ad);
        else if (ad.kind === "ui") uiNote(ad);
      }
      if (a.prompt && a.prompt.t !== A.promptT) { A.promptT = a.prompt.t; renderPrompt(a.prompt); }
      else if (!a.prompt) hideStageAsk();
      if (!opts.fresh && a.speak && A.speakOn) say(a.speak);
      if ($("#profile-dialog").open) renderProfile();
    }

    // ------------------------------------------------------------ persona chip
    function renderPersonaChip(p) {
      if (!p) return;
      $("#persona-name").textContent = p.archetype.name;
      const c = 2 * Math.PI * 15, ring = $("#persona-ring");
      ring.setAttribute("stroke-dasharray", `${(c * p.confidence).toFixed(1)} ${c.toFixed(1)}`);
      $("#persona-btn").title = `${p.archetype.name}: ${p.archetype.description} (${Math.round(100 * p.confidence)}% sure). Click to see or change it.`;
    }

    // ------------------------------------------------------------ loops bar
    function renderLoops(a) {
      const fired = new Set((a.adaptations || []).filter((x) => x.message || x.kind === "ui").map((x) => x.loop));
      const ol = clear($("#loops"));
      (a.loops || []).forEach((l, i) => {
        const li = h("li", { class: `loop ${fired.has(l.key) ? "fired" : ""} ${l.fired ? "used" : ""}`, tabindex: 0,
          title: `${l.answers}${l.last ? `\n\nLast: ${l.last}` : ""}` },
          h("span", { class: "loop-n mono", text: String(i + 1) }),
          h("span", { class: "loop-name", text: l.name }),
          h("span", { class: `loop-scale ${l.timescale}`, text: TIMESCALE[l.timescale] || l.timescale }),
          l.fired ? h("span", { class: "loop-count mono", text: String(l.fired) }) : "");
        li.addEventListener("click", openProfile);
        li.addEventListener("keydown", (e) => { if (e.key === "Enter") openProfile(); });
        ol.append(li);
      });
      const total = (a.loops || []).reduce((n, l) => n + l.fired, 0);
      $("#loops-sub").textContent = `${(a.loops || []).length} loops watch every change · ${total} adaptations so far`;
    }

    // ------------------------------------------------------------ notes from the loops
    const feed = $("#feed");
    function loopLabel(key) { return A.loopNames[key] || key; }
    function sysNote(ad) {
      const empty = $(".empty", feed); if (empty) empty.remove();
      feed.append(h("li", { class: `sys ${ad.kind}` }, h("small", { class: "sys-label", text: loopLabel(ad.loop) }),
        h("span", { text: ad.message })));
      P.scrollFeed();
    }
    function uiNote(ad) {
      feed.append(h("li", { class: "uinote" }, h("span", { text: ad.message }), ad.why ? h("small", { text: ` · ${ad.why}` }) : ""));
      P.scrollFeed();
    }

    // ------------------------------------------------------------ PaceSense asks
    function choiceButtons(prompt, onDone) {
      const box = h("div", { class: "choices" });
      for (const c of prompt.data.choices || []) {
        const b = h("button", { type: "button", class: "choice", text: c.label });
        b.addEventListener("click", async () => {
          $$("button", box).forEach((x) => { x.disabled = true; x.classList.toggle("picked", x === b); });
          onDone && onDone();
          await choose(c);
        });
        box.append(b);
      }
      return box;
    }
    function renderPrompt(pr) {
      const empty = $(".empty", feed); if (empty) empty.remove();
      const q = pr.data.question;
      const li = h("li", { class: `ask ${q ? "interview" : ""}` },
        h("small", { class: "sys-label", text: q ? "Guided interview" : `PaceSense asks · ${loopLabel(pr.loop)}` }),
        h("p", { class: "ask-text", text: pr.message }),
        pr.why ? h("p", { class: "ask-why", text: q ? q.why : `Because ${pr.why}.` }) : "",
        choiceButtons(pr, () => hideStageAsk()));
      feed.append(li); P.scrollFeed();
      // the same question, large, on the voice stage
      $("#stage-ask-text").textContent = pr.message;
      clear($("#stage-choices")).append(choiceButtons(pr, () => { $$(".ask:last-of-type .choice", feed).forEach((x) => { x.disabled = true; }); hideStageAsk(); }));
      $("#stage-ask").hidden = false;
    }
    function hideStageAsk() { $("#stage-ask").hidden = true; }

    async function choose(c) {
      try {
        if (c.say) return P.send(c.say, A.voiceMode ? "voice" : "typed");
        if (c.interview) return answerInterview(c.interview[0], c.interview[1], c.label);
        const act = c.action || "dismiss";
        if (act === "undo") return P.undo();
        if (act === "keep") { if (c.turn) await P.sendReaction(c.turn, "disagree", null, null); }
        if (act.startsWith("tab:")) { P.selectTab(`tab-${act.slice(4)}`, true); P.focusPanel("#panel-analysis"); }
        const r = await api("POST", `/sessions/${app.sid}/events`, { kind: "dismiss" });
        apply(r.adapt, { fresh: true });
      } catch (e) { toast(e.message, "error"); }
    }

    async function answerInterview(question, choice, label) {
      P.addSaid(label, A.voiceMode ? "voice" : "typed");
      P.setBusy(true);
      try {
        const r = await api("POST", `/sessions/${app.sid}/interview`, { question, choice });
        P.render(r.state);
        if (r.card && r.card.items.length) P.addCard(r.card);
        else if (r.reply && r.reply !== "Noted.") P.addReply(r.reply);
        apply(r.adapt);
        P.runSim();
      } catch (e) { toast(e.message, "error"); } finally { P.setBusy(false); }
    }

    async function startInterview() {
      if (!app.sid) return;
      try {
        const r = await api("POST", `/sessions/${app.sid}/interview/start`);
        apply(r.adapt);
        P.focusPanel("#panel-notes");
      } catch (e) { toast(e.message, "error"); }
    }

    // ------------------------------------------------------------ voice mode + stage
    async function setVoiceMode(on, { fromMic = false } = {}) {
      if (on === A.voiceMode && !fromMic) return;
      A.voiceMode = on;
      $("#talk-btn").setAttribute("aria-pressed", String(on));
      if (on && !fromMic && P.hasMic && !P.voiceOn()) P.startVoice();
      if (!on && P.voiceOn()) P.stopVoice(true);
      if (on && A.speakOn === null) A.speakOn = true;
      try {
        const r = await api("POST", `/sessions/${app.sid}/events`, { kind: on ? "voice_on" : "voice_off" });
        apply(r.adapt, { fresh: true });
      } catch (e) { toast(e.message, "error"); }
      if (on) $("#stage-status").textContent = P.hasMic ? "Listening… talk about her races" : "Voice mode (this browser has no speech recognition: type below)";
      if (on && !P.hasMic) $("#stage-input").focus();
    }
    function setStage(show) {
      $("#voice-stage").hidden = !show;
      if (!show && A.voiceMode && A.layout !== "voice") { A.voiceMode = false; $("#talk-btn").setAttribute("aria-pressed", "false"); }
    }
    function renderStage() {
      const st = app.st; if (!st || $("#voice-stage").hidden) return;
      $("#stage-target").textContent = st.prediction.finish;
      $("#stage-range").textContent = st.prediction.low_s ? `likely ${P.fmt.time(st.prediction.low_s, false)}–${P.fmt.time(st.prediction.high_s, false)}` : "";
      const box = clear($("#stage-races")), spot = new Set(app.spotlight || []);
      st.races.forEach((r, i) => {
        const bar = h("span", { class: "sr-bar" }, h("span", { class: "sr-fill" }));
        bar.firstChild.style.width = `${r.weight * 100}%`;
        const ev = r.evidence.direction > 0 ? ["warn", "slower than usual"] : r.evidence.direction < 0 ? ["ok", "faster than usual"] : ["neutral", "typical"];
        const card = h("button", { type: "button", class: `stage-race ${spot.has(i) ? "spot" : ""}`, title: `Start a sentence about ${r.event} ${r.year}` },
          h("span", { class: "sr-top" }, h("b", { text: `${r.event} ${r.year}` }), h("span", { class: "mono", text: r.finish })),
          h("span", { class: `chip ${ev[0]}`, text: ev[1] }),
          h("span", { class: "sr-w" }, bar, h("span", { class: "mono", text: r.weight.toFixed(1) })));
        card.addEventListener("click", () => { const inp = $("#stage-input"); inp.value = `${r.event} ${r.year} `; inp.focus(); });
        box.append(card);
      });
    }
    function onTranscript(text, final, finalText) {
      const t = $("#stage-transcript"), orb = $("#orb");
      if (final) { t.textContent = `“${finalText}”`; orb.classList.remove("hearing"); }
      else { t.textContent = text; orb.classList.add("hearing"); }
    }

    // ------------------------------------------------------------ speaking
    function readSpeakPref(ui) {
      const v = storage.get("pacesense.speak");
      return v === null ? !!ui.speak : v === "1";
    }
    function say(text) {
      if (!("speechSynthesis" in window) || !text) return;
      try {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1.04; u.lang = "en-US";
        u.onstart = () => { P.pauseListening(true); $("#orb").classList.add("speaking"); };
        u.onend = u.onerror = () => { P.pauseListening(false); $("#orb").classList.remove("speaking"); };
        speechSynthesis.speak(u);
      } catch { /* speech is optional */ }
    }

    // ------------------------------------------------------------ coach profile
    async function openProfile() {
      $("#profile-dialog").showModal();
      await renderProfile();
      api("POST", `/sessions/${app.sid}/events`, { kind: "panel", panel: "profile" }).catch(() => {});
    }
    async function renderProfile() {
      let prof;
      try { prof = await api("GET", `/coaches/${P.coachId()}/profile`); } catch (e) { toast(e.message, "error"); return; }
      const p = prof.persona;
      clear($("#arch")).append(
        h("div", { class: "arch-name" }, h("b", { text: p.archetype.name }), h("span", { class: "muted", text: ` · ${Math.round(100 * p.confidence)}% sure, from ${p.signals} signals` })),
        h("p", { text: p.archetype.description }));
      const dims = clear($("#dims"));
      for (const d of p.dimensions.concat(p.runner)) {
        const inp = h("input", { type: "range", min: 0, max: 1, step: 0.05, value: d.value, "aria-label": `${d.low} to ${d.high}` });
        inp.style.setProperty("--fill", `${d.value * 100}%`);
        inp.addEventListener("input", () => inp.style.setProperty("--fill", `${inp.value * 100}%`));
        inp.addEventListener("change", () => editProfile({ [d.key]: Number(inp.value) }));
        const marker = h("span", { class: "inferred", title: `Inferred: ${Math.round(100 * d.inferred)}%` });
        marker.style.left = `${d.inferred * 100}%`;
        dims.append(h("div", { class: "dim" },
          h("div", { class: "dim-labels" }, h("span", { text: d.low }), h("span", { text: d.high })),
          h("div", { class: "dim-track" }, inp, marker),
          h("div", { class: "dim-foot" },
            h("small", { class: "muted", text: d.override != null ? "Set by you" : d.cues.length ? `Because you ${d.cues[0].slice(2)}` : (d.key === "familiarity" ? "This runner only" : "No evidence yet") }),
            d.override != null ? h("button", { type: "button", class: "link-btn", text: "use what PaceSense learned", onclick: () => editProfile({ [d.key]: null }) }) : "")));
      }
      $("#layout-lock").value = prof.layout_lock || "";
      $("#interview-default").checked = !!prof.interview_default;
      renderRecord(prof.outcomes || []);
      const log = clear($("#why-log"));
      const entries = (A.adapt && A.adapt.log || []).slice().reverse();
      if (!entries.length) log.append(h("li", { class: "muted", text: "Nothing has adapted yet." }));
      entries.forEach((e) => log.append(h("li", null, h("b", { text: loopLabel(e.loop) }), " ", e.message,
        e.why ? h("small", { class: "muted", text: ` (because ${e.why})` }) : "")));
    }
    function renderRecord(out) {
      const box = clear($("#record"));
      if (!out.length) { box.append(h("p", { class: "muted", text: "Reveal the hidden race at the end of a demo session to start your track record." })); return; }
      const W = 420, H = 90, max = Math.max(0.5, ...out.map((o) => Math.abs(o.improvement)));
      const bw = Math.min(28, (W - 20) / out.length - 4);
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Accuracy change per revealed session" });
      svg.append(s("line", { x1: 0, x2: W, y1: H / 2, y2: H / 2, stroke: "var(--line-strong)" }));
      out.forEach((o, i) => {
        const hgt = (Math.abs(o.improvement) / max) * (H / 2 - 6), x = 10 + i * (bw + 4);
        svg.append(s("rect", { x, y: o.improvement > 0 ? H / 2 - hgt : H / 2, width: bw, height: Math.max(hgt, 1), rx: 2,
          fill: o.improvement > 0.05 ? "var(--good)" : o.improvement < -0.05 ? "var(--bad)" : "var(--past)" }));
      });
      const k = out.filter((o) => o.improvement > 0.05).length;
      box.append(svg, h("p", { class: "muted", text: `Your changes improved accuracy in ${k} of ${out.length} sessions (bars above the line: more accurate).` }));
    }
    async function editProfile(changes) {
      try {
        const r = await api("PUT", `/coaches/${P.coachId()}/profile`, { ...changes, session_id: app.sid });
        apply(r.adapt, { fresh: true });
      } catch (e) { toast(e.message, "error"); }
    }

    // ------------------------------------------------------------ hooks + wiring
    P.hooks.apply = apply;
    P.hooks.transcript = onTranscript;
    P.hooks.voice = (on) => setVoiceMode(on, { fromMic: true });
    P.hooks.panel = (panel) => {
      if (!app.sid) return;
      api("POST", `/sessions/${app.sid}/events`, { kind: "panel", panel }).then((r) => apply(r.adapt, { fresh: true })).catch(() => {});
    };

    $("#talk-btn").addEventListener("click", () => setVoiceMode(!A.voiceMode));
    $("#stage-close").addEventListener("click", () => setVoiceMode(false));
    $("#guide-btn").addEventListener("click", startInterview);
    $("#persona-btn").addEventListener("click", openProfile);
    $("#why-btn").addEventListener("click", openProfile);
    $("#speak-btn").addEventListener("click", () => {
      A.speakOn = !A.speakOn; storage.set("pacesense.speak", A.speakOn ? "1" : "0");
      $("#speak-btn").setAttribute("aria-pressed", String(A.speakOn));
      if (!A.speakOn && "speechSynthesis" in window) speechSynthesis.cancel();
      toast(A.speakOn ? "Spoken replies on" : "Spoken replies off");
    });
    $("#stage-form").addEventListener("submit", (e) => {
      e.preventDefault(); const v = $("#stage-input").value.trim(); if (!v) return;
      $("#stage-input").value = ""; onTranscript("", true, v); P.send(v, "typed");
    });
    $("#layout-lock").addEventListener("change", (e) => editProfile({ layout_lock: e.target.value || null }));
    $("#interview-default").addEventListener("change", (e) => editProfile({ interview_default: e.target.checked }));
    $("#persona-reset").addEventListener("click", () => editProfile({ reset: true }));
    $("#persona-forget").addEventListener("click", async () => {
      const b = $("#persona-forget");
      if (b.dataset.confirm !== "1") { b.dataset.confirm = "1"; b.textContent = "Click again to forget everything"; return; }
      try {
        await api("DELETE", `/coaches/${P.coachId()}/profile`);
        $("#profile-dialog").close(); toast("Everything learned about you was deleted.");
        A.layout = null; P.openRunner("demo");
      } catch (e) { toast(e.message, "error"); }
      b.dataset.confirm = ""; b.textContent = "Forget me completely";
    });
    document.addEventListener("keydown", (e) => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName) && document.activeElement.type !== "range";
      if (typing || e.ctrlKey || e.metaKey || e.altKey || document.querySelector("dialog[open]")) return;
      if (e.key === "v" || e.key === "V") { e.preventDefault(); setVoiceMode(!A.voiceMode); }
      else if (e.key === "g" || e.key === "G") { e.preventDefault(); startInterview(); }
      else if (e.key === "Escape" && A.voiceMode) setVoiceMode(false);
    });
  }
})();
