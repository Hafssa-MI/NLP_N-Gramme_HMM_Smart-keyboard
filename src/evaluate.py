import json
import math
import random
import time

from src.ngram import UNK, BOS, AddK
from src.engine import Predictor


def positions(sentences, limit, seed=0):
    pos = [(s, i) for s in sentences for i in range(len(s) + 1)]
    random.Random(seed).shuffle(pos)
    return pos[:limit]


def target(s, i):
    return s[i] if i < len(s) else "</s>"


def scan(pred, pos, betas):
    """One pass over positions; log-likelihood and rank metrics for every beta (beta = HMM weight)."""
    res = {b: dict(ll=0.0, n=0, h1=0, h3=0, h5=0, rr=0.0, inv=0) for b in betas}
    for s, i in pos:
        ctx = s[:i]
        wid = pred.vocab.stoi.get(target(s, i), UNK)
        base, _, _, f = pred._parts(ctx)
        for b in betas:
            p = pred._combine(base, f, True, b)
            r = res[b]
            r["ll"] += math.log(p[wid] / p.sum())
            r["n"] += 1
            if wid != UNK:
                q = p.copy()
                q[UNK] = 0.0
                q[BOS] = 0.0
                rank = int((q > q[wid]).sum()) + 1
                r["inv"] += 1
                r["h1"] += rank <= 1
                r["h3"] += rank <= 3
                r["h5"] += rank <= 5
                r["rr"] += 1.0 / rank
    return res


def summarize(r):
    return {"perplexity": math.exp(-r["ll"] / r["n"]), "top1": r["h1"] / r["inv"],
            "top3": r["h3"] / r["inv"], "top5": r["h5"] / r["inv"], "mrr": r["rr"] / r["inv"]}


def keystroke_savings(pred, pos, use_hmm, k=3, n=800):
    """Share of keystrokes saved if the user taps a suggestion as soon as it appears (space counts as 1)."""
    saved = total = 0
    for s, i in pos[:n]:
        if i >= len(s) or s[i] not in pred.vocab.stoi:
            continue
        w, full = s[i], len(s[i]) + 1
        typed = full
        for p in range(len(w)):
            if w in pred.topk_words(s[:i], w[:p], k, use_hmm):
                typed = p + 1
                break
        saved += full - min(typed, full)
        total += full
    return saved / total


def addk_ppl(model, pos, vocab):
    ll = 0.0
    for s, i in pos:
        ctx = vocab.encode(s[max(0, i - 2):i])
        ll += math.log(model.prob(ctx, vocab.stoi.get(target(s, i), UNK)))
    return math.exp(-ll / len(pos))


def run(train, val, test, vocab, kn, hmm, hmm_eval, val_n=6000, test_n=15000, out="results/metrics.json"):
    pred = Predictor(kn, hmm, vocab, beta=0.0)
    vp, tp = positions(val, val_n, 1), positions(test, test_n, 2)

    # 1) tune the HMM weight on VALIDATION only
    betas = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    v = scan(pred, vp, betas)
    sweep = [{"beta": b, **summarize(v[b])} for b in betas]
    best = min(sweep, key=lambda r: r["perplexity"])["beta"]
    pred.beta = best

    # 2) add-k baselines, k tuned on validation
    models = []
    for n in (1, 2, 3):
        m = AddK(train, vocab, n, 1.0)
        scores = {}
        for k in (0.001, 0.01, 0.05, 0.1, 0.5, 1.0):
            m.k = k
            scores[k] = addk_ppl(m, vp, vocab)
        m.k = min(scores, key=scores.get)
        models.append({"name": f"{['Unigram', 'Bigram', 'Trigram'][n - 1]}, add-k (k={m.k})",
                       "perplexity": addk_ppl(m, tp, vocab), "top1": None, "top3": None,
                       "top5": None, "mrr": None, "kss": None})

    # 3) final test report (once)
    t = scan(pred, tp, [0, best])
    kss0 = keystroke_savings(pred, tp, use_hmm=False)
    kss1 = keystroke_savings(pred, tp, use_hmm=True)
    models.append({"name": "Kneser-Ney trigram", **summarize(t[0]), "kss": kss0})
    models.append({"name": f"Kneser-Ney trigram + HMM (beta={best})", **summarize(t[best]), "kss": kss1})

    t0 = time.perf_counter()
    for s, i in tp[:300]:
        pred.suggest(" ".join(s[:i]) + " ", 3)
    latency = (time.perf_counter() - t0) / 300 * 1000

    n_test = sum(len(s) for s in test)
    metrics = {
        "dataset": {"name": __import__("config").DATASET.get("name", "csv"),
                    "train_tokens": sum(len(s) for s in train), "test_tokens": n_test,
                    "vocab": vocab.V,
                    "oov_rate": sum(w not in vocab.stoi for s in test for w in s) / n_test,
                    "eval_positions": len(tp)},
        "models": models, "beta_sweep": sweep, "best_beta": best, "hmm": hmm_eval,
        "latency_ms": latency,
    }
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics
