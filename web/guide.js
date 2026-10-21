
(() => {
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const overlayMode = () => matchMedia("(max-width: 1100px)").matches;
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
};

/* ------------------------------------------------------------------ content */
const GUIDE = [
  { title: "Quick start", open: true, type: "steps", rows: [
    "Open the Writer view and type in the Compose box. Suggestions update after every keystroke.",
    "Press Tab to accept the first suggestion, or click any suggestion.",
    "Click a phrase completion to insert several words at once.",
    "Open Analysis to see how the tagger reads a sentence and how the two models differ.",
    "Open Evaluation to see how each model scores on text it was never trained on.",
  ]},
  { title: "Writer view", type: "items",
    intro: "Select Show to highlight the element on the page.",
    rows: [
    { name: "Compose box", view: "writer", target: "#editor",
      text: "Type here. The engine reads the text before the caret, so you can move the caret and get suggestions for any position. While you type a word it is completed; after a space the next word is predicted." },
    { name: "Suggestion slots", view: "writer", target: "#slots",
      text: "The first slot, shaded blue, is the most probable word. The small label is its predicted part of speech and the bar compares probabilities. A green edge means your personal profile boosted the word. \"end of sentence\" means the sentence is likely to finish here." },
    { name: "Phrase completions", view: "writer", target: "#phrases", scope: "card",
      text: "Four-word continuations found by beam search. Click one to insert it. \"ends sentence\" marks a phrase that finishes the sentence." },
    { name: "Engine state", view: "writer", target: "#stMode", scope: "card",
      text: "Mode shows whether the engine is predicting a new word or completing the one you are typing. Context order is the n-gram level that had data for your last words: trigram is the most specific, unigram the fallback. Latency is the time for one request. Red chips are words outside the training vocabulary." },
    { name: "Next tag, HMM", view: "writer", target: "#nextTags", scope: "card",
      text: "The part-of-speech tags the HMM expects at the next position, with probabilities. This distribution reweights the n-gram candidates so that grammatical words rank higher." },
    { name: "Syntactic model switch", view: "writer", target: "#optHmm", scope: "parent",
      text: "Turns the HMM reweighting on or off. Type the same sentence with the switch on and off to see what the HMM changes." },
    { name: "Personal profile switch", view: "writer", target: "#optPersonal", scope: "parent",
      text: "Turns the personal profile on or off in the suggestions without deleting it." },
    { name: "Number of suggestions", view: "writer", target: "#seg",
      text: "Show 3 or 5 suggestions. Alt+4 and Alt+5 only work when five are shown." },
    { name: "Personal profile", view: "writer", target: "#btnLearn", scope: "parent",
      text: "Learn from text reads what is in the Compose box and remembers its words and phrasing, including names and slang the model has never seen. Reset deletes the profile. It is kept in server memory for this session only and is never written to disk." },
  ]},
  { title: "Keyboard shortcuts", type: "keys", rows: [
    { keys: ["Tab"], text: "Accept the first suggestion" },
    { keys: ["Alt", "1 to 5"], text: "Accept suggestion number 1 to 5" },
    { keys: ["Esc"], text: "Close this guide" },
  ]},
  { title: "Analysis view", type: "items", rows: [
    { name: "Sentence box", view: "analysis", target: "#analysisInput",
      text: "Type any sentence. A trailing space asks for the next word; a partial word asks for a completion." },
    { name: "Tagged words", view: "analysis", target: "#tagRow",
      text: "Each word with the tag chosen by Viterbi decoding, which finds the most likely tag sequence for the whole sentence." },
    { name: "Tag legend", view: "analysis", target: "#legend",
      text: "The tag set used by the HMM (universal part-of-speech tags) with the colour of each." },
    { name: "N-gram only", view: "analysis", target: "#rankBase", scope: "card",
      text: "Candidates ranked by the Kneser-Ney trigram model alone, with their probabilities." },
    { name: "N-gram with HMM", view: "analysis", target: "#rankHmm", scope: "card",
      text: "The same candidates after syntactic reweighting. Compare the two lists to see which words the grammar model promotes or demotes." },
  ]},
  { title: "Evaluation view", type: "items", rows: [
    { name: "Summary cards", view: "evaluation", target: "#metricCards",
      text: "Headline numbers for the final model on held-out test documents: perplexity, top-3 accuracy, keystroke savings, tagging accuracy and latency." },
    { name: "Model comparison", view: "evaluation", target: "#modelTable", scope: "card",
      text: "Every model on the same test text. The best value in each column is blue. A dash means the metric was not measured for that model; the add-k baselines report perplexity only." },
    { name: "HMM weight", view: "evaluation", target: "#sweepTable", scope: "card",
      text: "How the HMM weight (beta) affects results. It was chosen on a validation split, never on the test set. Beta 0 means the HMM is switched off." },
    { name: "POS tagger", view: "evaluation", target: "#posStats", scope: "card",
      text: "Accuracy of the HMM tagger against a most-frequent-tag baseline, split into known and unknown words." },
  ]},
  { title: "Terms used here", type: "terms", rows: [
    ["N-gram model", "Predicts the next word from the previous two words using counts from the training text. Smoothing makes sure unseen word sequences still get some probability."],
    ["Kneser-Ney", "The smoothing method used here. Lower orders count how many different words a word follows, not how often it appears."],
    ["HMM", "A hidden Markov model over part-of-speech tags. It knows which tag types tend to follow each other."],
    ["Viterbi", "The algorithm that finds the single best tag sequence for a sentence."],
    ["Perplexity", "How surprised the model is by real text, roughly the number of equally likely words it is choosing between at each step. Lower is better."],
    ["Top-1, top-3, top-5", "Share of positions where the true next word is among the first 1, 3 or 5 suggestions."],
    ["MRR", "Mean reciprocal rank. 1.0 would mean the correct word is always the first suggestion."],
    ["Keystroke savings", "Share of keystrokes saved if you tap a suggestion as soon as it appears."],
    ["Beta", "Weight of the HMM reweighting. 0 means n-gram only."],
    ["Unknown word", "A word outside the training vocabulary. It is shown in red and handled as an unknown token."],
  ]},
];

/* ------------------------------------------------------------------ dom helpers */
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

let flashTimer;
function reveal(r) {
  const tab = $(`.tab[data-view="${r.view}"]`);
  const pane = $("#view-" + r.view);
  const switching = !!pane && !pane.classList.contains("is-active");
  if (switching && tab) tab.click();
  if (overlayMode()) setOpen(false);
  setTimeout(() => {
    let t = $(r.target);
    if (!t) return;
    if (r.scope === "card") t = t.closest(".card") || t;
    else if (r.scope === "parent") t = t.parentElement || t;
    $$(".g-flash").forEach(x => x.classList.remove("g-flash"));
    t.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" });
    void t.offsetWidth;
    t.classList.add("g-flash");
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => t.classList.remove("g-flash"), 2400);
  }, switching ? 420 : 60);
}

function buildSection(sec) {
  const d = el("details", "g-sec");
  if (sec.open) d.open = true;
  d.append(el("summary", null, sec.title));
  const c = el("div", "g-content");
  if (sec.intro) c.append(el("p", "g-intro", sec.intro));

  if (sec.type === "steps") {
    const ol = el("ol", "g-steps");
    sec.rows.forEach(t => ol.append(el("li", null, t)));
    c.append(ol);
  } else if (sec.type === "items") {
    const ul = el("ul", "g-list");
    sec.rows.forEach(r => {
      const li = el("li", "g-item");
      const txt = el("div", "g-text");
      txt.append(el("strong", null, r.name), el("p", null, r.text));
      const b = el("button", "g-show", "Show");
      b.type = "button";
      b.setAttribute("aria-label", "Show " + r.name + " on the page");
      b.addEventListener("click", () => reveal(r));
      li.append(txt, b);
      ul.append(li);
    });
    c.append(ul);
  } else if (sec.type === "keys") {
    const dl = el("dl", "g-keys");
    sec.rows.forEach(r => {
      const row = el("div"), dt = el("dt");
      r.keys.forEach((k, i) => { if (i) dt.append(document.createTextNode(" + ")); dt.append(el("kbd", null, k)); });
      row.append(dt, el("dd", null, r.text));
      dl.append(row);
    });
    c.append(dl);
  } else {
    const dl = el("dl", "g-terms");
    sec.rows.forEach(([t, x]) => { const row = el("div"); row.append(el("dt", null, t), el("dd", null, x)); dl.append(row); });
    c.append(dl);
  }
  d.append(c);
  return d;
}

/* ------------------------------------------------------------------ build */
const nav = $("#tabs");
if (!nav) return;

const btn = el("button", "guide-btn");
btn.type = "button";
btn.id = "guideBtn";
btn.setAttribute("aria-pressed", "false");
btn.setAttribute("aria-controls", "guide");
btn.innerHTML = '<span>Guide</span><span class="gb-track" aria-hidden="true"></span>';
nav.append(btn);

const backdrop = el("div", "g-backdrop");
const drawer = el("aside", "guide");
drawer.id = "guide";
drawer.setAttribute("aria-label", "Interface guide");
drawer.setAttribute("aria-hidden", "true");

const head = el("div", "g-head");
const title = el("div");
title.append(el("h2", null, "Interface guide"), el("p", null, "How each part of the keyboard works"));
const closeBtn = el("button", "g-close", "Close");
closeBtn.type = "button";
head.append(title, closeBtn);

const body = el("div", "g-body");
GUIDE.forEach(s => body.append(buildSection(s)));
drawer.append(head, body);
document.body.append(backdrop, drawer);

/* ------------------------------------------------------------------ open / close */
function setOpen(on, focus = true) {
  document.body.classList.toggle("guide-open", on);
  btn.setAttribute("aria-pressed", String(on));
  drawer.setAttribute("aria-hidden", String(!on));
  if (on) {
    store.set("sk_guide_seen", "1");
    if (focus) setTimeout(() => closeBtn.focus({ preventScroll: true }), 60);
  }
}

btn.addEventListener("click", () => setOpen(!document.body.classList.contains("guide-open")));
closeBtn.addEventListener("click", () => { setOpen(false); btn.focus(); });
backdrop.addEventListener("click", () => setOpen(false));
addEventListener("keydown", e => {
  if (e.key === "Escape" && document.body.classList.contains("guide-open")) { setOpen(false); btn.focus(); }
});

/* first visit on a wide screen: open the guide once so the interface is explained */
if (!store.get("sk_guide_seen") && innerWidth > 1200) setTimeout(() => setOpen(true, false), 700);
})();