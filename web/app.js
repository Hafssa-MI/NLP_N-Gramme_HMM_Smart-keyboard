(() => {
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
};
let uid = store.get("sk_uid");
if (!uid) { uid = crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2); store.set("sk_uid", uid); }

async function call(path, body, signal) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  if (!res.ok) throw new Error(path + " " + res.status);
  return res.json();
}
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const pct = (x, d = 1) => (x == null ? "-" : (x * 100).toFixed(d) + "%");
const num = (x, d = 2) => (x == null ? "-" : x.toFixed(d));

const TAG_NAMES = { NOUN: "Noun", VERB: "Verb", ADJ: "Adjective", ADV: "Adverb", DET: "Determiner",
  ADP: "Preposition", PRON: "Pronoun", CONJ: "Conjunction", PRT: "Particle", NUM: "Number", X: "Other", END: "Sentence end" };

function tagEl(tag) {
  const s = document.createElement("span");
  s.className = "tag"; s.textContent = tag; s.dataset.tag = tag;
  return s;
}

function setStatus(ok, text) {
  const el = $("#status");
  el.classList.toggle("is-ok", ok); el.classList.toggle("is-bad", !ok);
  $("#statusText").textContent = text || (ok ? "Model ready" : "Offline");
}

/* ---------- tabs ---------- */
const tabs = $$(".tab"), ink = $("#tabInk");
function moveInk(el) { ink.style.width = el.offsetWidth + "px"; ink.style.transform = `translateX(${el.offsetLeft}px)`; }
function show(view) {
  const tab = tabs.find(t => t.dataset.view === view) || tabs[0];
  tabs.forEach(t => t.classList.toggle("is-active", t === tab));
  $$(".view").forEach(v => v.classList.toggle("is-active", v.id === "view-" + tab.dataset.view));
  moveInk(tab);
  if (tab.dataset.view === "analysis") runAnalysis();
  if (tab.dataset.view === "evaluation") loadMetrics();
  history.replaceState(null, "", "#" + tab.dataset.view);
}
tabs.forEach(t => t.addEventListener("click", () => show(t.dataset.view)));
addEventListener("resize", () => moveInk($(".tab.is-active")));

/* ---------- rank list helper (animated bars) ---------- */
function renderRank(ul, items) {
  ul.innerHTML = "";
  const max = Math.max(0.0001, ...items.map(i => i.value));
  items.forEach((it, n) => {
    const li = document.createElement("li");
    li.style.setProperty("--i", n);
    const label = document.createElement("span");
    label.className = "rank-label";
    label.append(document.createTextNode(it.label));
    if (it.tag) label.append(tagEl(it.tag));
    const bar = document.createElement("span"); bar.className = "bar";
    const fill = document.createElement("i"); bar.append(fill);
    const val = document.createElement("span"); val.className = "rank-val"; val.textContent = pct(it.value);
    li.append(label, bar, val); ul.append(li);
    requestAnimationFrame(() => requestAnimationFrame(() => fill.style.setProperty("--w", (it.value / max).toFixed(3))));
  });
}

/* ---------- writer ---------- */
const state = { k: 3, hmm: true, personal: true, seq: 0, current: [] };
const editor = $("#editor"), slotsEl = $("#slots"), phrasesEl = $("#phrases");
let slotEls = [], ctrl = null;

function buildSlots() {
  slotsEl.innerHTML = ""; slotEls = [];
  for (let i = 0; i < state.k; i++) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "slot is-empty";
    b.innerHTML = '<span class="slot-top"><span class="slot-word"></span><span class="tag" hidden></span></span>' +
      '<span class="bar"><i></i></span>' +
      `<span class="slot-meta"><span class="slot-prob"></span><span>${i === 0 ? "Tab" : "Alt+" + (i + 1)}</span></span>`;
    b.addEventListener("mousedown", e => e.preventDefault());   // keep the caret in the editor
    b.addEventListener("click", () => accept(i));
    slotsEl.append(b); slotEls.push(b);
  }
}

function renderSlots(list) {
  state.current = list;
  slotEls.forEach((el, i) => {
    const s = list[i], w = $(".slot-word", el), t = $(".tag", el);
    el.classList.toggle("is-empty", !s);
    el.classList.toggle("is-personal", !!(s && s.personal));
    if (!s) { w.textContent = ""; t.hidden = true; $(".slot-prob", el).textContent = ""; $(".bar i", el).style.setProperty("--w", 0); return; }
    const label = s.eos ? "end of sentence" : s.word;
    if (w.textContent !== label) {
      w.textContent = label;
      if (!reduce) w.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 200, easing: "ease-out" });
    }
    t.textContent = s.pos; t.dataset.tag = s.pos; t.hidden = false;
    $(".slot-prob", el).textContent = pct(s.prob);
    $(".bar i", el).style.setProperty("--w", Math.min(1, s.prob / Math.max(list[0].prob, 0.25)).toFixed(3));
  });
}

function renderPhrases(list) {
  phrasesEl.innerHTML = "";
  list.forEach((c, n) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button"; b.style.setProperty("--i", n);
    b.innerHTML = `<span></span><span class="ended">${c.ended ? "ends sentence" : ""}</span>`;
    b.firstChild.textContent = c.text;
    b.addEventListener("mousedown", e => e.preventDefault());
    b.addEventListener("click", () => insertAtCaret(c.text, c.ended));
    li.append(b); phrasesEl.append(li);
  });
}

function renderState(s) {
  $("#stMode").textContent = s.mode === "completion" ? `Completing "${s.prefix}"` : "Next word";
  $("#stOrder").textContent = s.order;
  $("#stLatency").textContent = s.latency_ms.toFixed(1) + " ms";
  const ctx = $("#stContext"); ctx.innerHTML = "";
  (s.context.length ? s.context : ["start of sentence"]).forEach((w, i) => {
    const c = document.createElement("span");
    c.className = "chip" + (s.oov.includes(w) ? " is-oov" : ""); c.textContent = w;
    c.title = s.oov.includes(w) ? "Not in the training vocabulary" : ""; c.style.animationDelay = i * 30 + "ms";
    ctx.append(c);
  });
  renderRank($("#nextTags"), s.next_tags.map(t => ({ label: TAG_NAMES[t.tag] || t.tag, value: t.p, tag: null })));
}

async function refresh() {
  const my = ++state.seq;
  if (ctrl) ctrl.abort();
  ctrl = new AbortController();
  const text = editor.value.slice(0, editor.selectionStart);
  $("#wordCount").textContent = (editor.value.trim() ? editor.value.trim().split(/\s+/).length : 0) + " words";
  try {
    const [s, c] = await Promise.all([
      call("/api/suggest", { text, k: state.k, user_id: uid, use_hmm: state.hmm, personalize: state.personal }, ctrl.signal),
      call("/api/complete", { text, length: 4, beam: 3, use_hmm: state.hmm }, ctrl.signal),
    ]);
    if (my !== state.seq) return;
    renderSlots(s.suggestions); renderPhrases(c.completions); renderState(s); setStatus(true);
  } catch (e) { if (e.name !== "AbortError") setStatus(false); }
}

function insertAtCaret(insert, endsSentence = false) {
  const pos = editor.selectionStart, v = editor.value;
  const before = v.slice(0, pos), after = v.slice(pos);
  const m = before.match(/[A-Za-z']+$/);
  let head = m ? before.slice(0, before.length - m[0].length) : before;
  if (/(^|[.!?]\s*)$/.test(head)) insert = insert.charAt(0).toUpperCase() + insert.slice(1);
  const added = head + insert + (endsSentence ? ". " : " ");
  editor.value = added + after.replace(/^ /, "");
  editor.setSelectionRange(added.length, added.length);
  editor.focus(); refresh();
}

function accept(i) {
  const s = state.current[i];
  if (!s) return;
  if (s.eos) {
    const pos = editor.selectionStart, before = editor.value.slice(0, pos).replace(/\s+$/, "");
    const added = before + ". ";
    editor.value = added + editor.value.slice(pos).replace(/^\s+/, "");
    editor.setSelectionRange(added.length, added.length); editor.focus(); refresh();
  } else insertAtCaret(s.word);
}

editor.addEventListener("input", debounce(refresh, 70));
editor.addEventListener("click", debounce(refresh, 30));
editor.addEventListener("keyup", e => { if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) refresh(); });
editor.addEventListener("keydown", e => {
  if (e.key === "Tab" && !e.shiftKey && state.current[0]) { e.preventDefault(); accept(0); }
  else if (e.altKey && /^Digit[1-5]$/.test(e.code)) { e.preventDefault(); accept(+e.code.slice(5) - 1); }
});

$("#optHmm").addEventListener("change", e => { state.hmm = e.target.checked; refresh(); });
$("#optPersonal").addEventListener("change", e => { state.personal = e.target.checked; refresh(); });
$$("#seg button").forEach(b => b.addEventListener("click", () => {
  $$("#seg button").forEach(x => x.classList.toggle("is-active", x === b));
  state.k = +b.dataset.k; buildSlots(); refresh();
}));

$("#btnLearn").addEventListener("click", async () => {
  const text = editor.value.trim();
  if (!text) { $("#profileNote").textContent = "Type some text first."; return; }
  try {
    const r = await call("/api/learn", { text, user_id: uid });
    $("#profileNote").textContent = `Learned ${r.tokens} words, ${r.n_new_words} new` +
      (r.new_words.length ? ` (${r.new_words.slice(0, 5).join(", ")})` : "") + `. Profile: ${r.profile_tokens} words.`;
    refresh();
  } catch { $("#profileNote").textContent = "Could not reach the server."; }
});
$("#btnForget").addEventListener("click", async () => {
  await call("/api/forget", { user_id: uid }).catch(() => {});
  $("#profileNote").textContent = "Profile reset."; refresh();
});

/* ---------- analysis ---------- */
async function runAnalysis() {
  const text = $("#analysisInput").value;
  try {
    const [plain, withHmm] = await Promise.all([
      call("/api/suggest", { text, k: 6, use_hmm: false, personalize: false }),
      call("/api/suggest", { text, k: 6, use_hmm: true, personalize: false }),
    ]);
    const row = $("#tagRow"); row.innerHTML = "";
    withHmm.tags.forEach((t, i) => {
      const d = document.createElement("div"); d.className = "tok"; d.style.setProperty("--i", i);
      const w = document.createElement("span"); w.textContent = t.word;
      d.append(w, tagEl(t.tag)); row.append(d);
    });
    if (!withHmm.tags.length) row.innerHTML = '<span class="muted">Type a sentence to see its part-of-speech tags.</span>';
    const toItems = r => r.suggestions.map(s => ({ label: s.eos ? "end of sentence" : s.word, value: s.prob, tag: s.pos }));
    renderRank($("#rankBase"), toItems(plain));
    renderRank($("#rankHmm"), toItems(withHmm));
  } catch { setStatus(false); }
}
$("#analysisInput").addEventListener("input", debounce(runAnalysis, 120));

/* ---------- evaluation ---------- */
let metrics = null;
function countUp(el, to, fmt, ms = 900) {
  if (reduce) { el.textContent = fmt(to); return; }
  const t0 = performance.now();
  const step = t => { const k = Math.min(1, (t - t0) / ms); el.textContent = fmt(to * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

async function loadMetrics() {
  if (metrics) return;
  try { metrics = await call("/api/metrics"); }
  catch { $("#metricCards").innerHTML = '<p class="muted">Run the evaluation step to create results/metrics.json.</p>'; return; }
  const rows = metrics.models, last = rows[rows.length - 1], d = metrics.dataset;

  const cards = [
    ["Perplexity", last.perplexity, v => v.toFixed(1)],
    ["Top-3 accuracy", last.top3 * 100, v => v.toFixed(1) + "%"],
    ["Keystroke savings", last.kss * 100, v => v.toFixed(1) + "%"],
    ["POS tagging accuracy", metrics.hmm.accuracy * 100, v => v.toFixed(1) + "%"],
    ["Latency per request", metrics.latency_ms, v => v.toFixed(1) + " ms"],
  ];
  const wrap = $("#metricCards"); wrap.innerHTML = "";
  cards.forEach(([label, value, fmt], i) => {
    const m = document.createElement("div"); m.className = "metric"; m.style.setProperty("--i", i);
    m.innerHTML = `<div class="label">${label}</div><div class="value">0</div>`;
    wrap.append(m); countUp($(".value", m), value, fmt);
  });

  $("#datasetLine").textContent = `${d.name} | ${d.train_tokens.toLocaleString()} training tokens | vocabulary ${d.vocab.toLocaleString()} | ${pct(d.oov_rate)} unknown words in test`;

  const cols = [["perplexity", "Perplexity", v => num(v), "min"], ["top1", "Top-1", v => pct(v), "max"],
    ["top3", "Top-3", v => pct(v), "max"], ["top5", "Top-5", v => pct(v), "max"],
    ["mrr", "MRR", v => num(v, 3), "max"], ["kss", "Keystroke savings", v => pct(v), "max"]];
  const best = {};
  cols.forEach(([k, , , dir]) => {
    const vals = rows.map(r => r[k]).filter(v => v != null);
    best[k] = dir === "min" ? Math.min(...vals) : Math.max(...vals);
  });
  $("#modelTable").innerHTML = "<thead><tr><th>Model</th>" + cols.map(c => `<th>${c[1]}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map(r => `<tr><td>${r.name}</td>` + cols.map(([k, , f]) =>
      `<td class="${r[k] != null && r[k] === best[k] ? "best" : ""}">${r[k] == null ? "-" : f(r[k])}</td>`).join("") + "</tr>").join("") + "</tbody>";

  const sw = metrics.beta_sweep, maxT = Math.max(...sw.map(s => s.top3));
  $("#sweepTable").innerHTML = "<thead><tr><th>HMM weight</th><th>Perplexity</th><th>Top-3</th></tr></thead><tbody>" +
    sw.map(s => `<tr><td>${s.beta}${s.beta === metrics.best_beta ? " (selected)" : ""}</td><td class="${s.beta === metrics.best_beta ? "best" : ""}">${num(s.perplexity)}</td>` +
      `<td>${pct(s.top3)}<span class="bar cellbar"><i style="--w:${(s.top3 / maxT).toFixed(3)}"></i></span></td></tr>`).join("") + "</tbody>";

  const h = metrics.hmm;
  $("#posStats").innerHTML = [["Accuracy (Viterbi)", pct(h.accuracy)], ["Most-frequent-tag baseline", pct(h.baseline)],
    ["Known words", pct(h.accuracy_known)], ["Unknown words", pct(h.accuracy_unknown)],
    ["Unknown word share", pct(h.unknown_share)], ["Held-out tokens", h.n_tokens.toLocaleString()]]
    .map(([a, b]) => `<div><dt>${a}</dt><dd>${b}</dd></div>`).join("");
}

/* ---------- init ---------- */
(async function init() {
  buildSlots();
  try {
    const info = await call("/api/info");
    setStatus(true, "Model ready");
    const legend = $("#legend");
    info.tags.concat("END").forEach(t => { const s = tagEl(t); s.textContent = TAG_NAMES[t] || t; legend.append(s); });
  } catch { setStatus(false); }
  show((location.hash || "#writer").slice(1));
  refresh();
})();
})();
