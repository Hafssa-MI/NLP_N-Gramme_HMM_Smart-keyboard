import math
import re
import time
from collections import Counter

import numpy as np

from src.data import tokenize, split_sentences
from src.ngram import UNK, EOS, BOS

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
LEVELS = {1: "unigram", 2: "bigram", 3: "trigram"}


def parse(text):
    """-> (completed words of the current sentence, prefix of the word being typed)."""
    cur = re.split(r"[.!?\n]+", text.lower().replace("\u2019", "'"))[-1]
    toks = WORD.findall(cur)
    if toks and (cur[-1].isalpha() or cur[-1] == "'"):
        return toks[:-1], toks[-1]
    return toks, ""


class UserCache:
    """Per-user adaptive model over WORD STRINGS, so words outside the vocabulary (names, slang) are learned too."""
    MAX_SHARE, PRIOR = 0.5, 3.0

    def __init__(self):
        self.tri, self.bi, self.uni = {}, {}, Counter()
        self.n_tokens = 0

    def learn(self, text, vocab):
        new, n = set(), 0
        for sent in split_sentences(text):
            hist = ["<s>", "<s>"]
            for w in tokenize(sent):
                self.tri.setdefault((hist[-2], hist[-1]), Counter())[w] += 1
                self.bi.setdefault(hist[-1], Counter())[w] += 1
                self.uni[w] += 1
                if w not in vocab.stoi:
                    new.add(w)
                hist.append(w)
                n += 1
        self.n_tokens += n
        return {"tokens": n, "n_new_words": len(new), "new_words": sorted(new)[:12],
                "profile_tokens": self.n_tokens}

    def blend(self, p, ctx, prefix, vocab):
        u, v = (["<s>", "<s>"] + ctx)[-2:]
        sources = [self.tri.get((u, v)), self.bi.get(v)]
        if prefix:
            sources.append(self.uni)
        cnt, tot, src = None, 0, None
        for cand in sources:
            if not cand:
                continue
            sel = {w: c for w, c in cand.items() if w.startswith(prefix)}
            if sel:
                cnt, tot, src = sel, sum(sel.values()), cand
                break
        if not cnt:
            return p, [], set()
        m = min(self.MAX_SHARE, tot / (tot + self.PRIOR)) * (0.5 if src is self.uni else 1.0)
        q, extra = p * (1 - m), []
        for w, c in cnt.items():
            i = vocab.stoi.get(w)
            if i is None:
                extra.append((w, m * c / tot))
            else:
                q[i] += m * c / tot
        z = q.sum() + sum(x for _, x in extra)
        if z > 0:
            q = q / z
            extra = [(w, x / z) for w, x in extra]
        return q, extra, set(cnt)


class Predictor:
    def __init__(self, kn, hmm, vocab, beta=0.4):
        self.kn, self.hmm, self.vocab, self.beta = kn, hmm, vocab, beta
        self.F = hmm.vocab_matrix(vocab)

    # ---- distribution pieces
    def _parts(self, ctx):
        base, level = self.kn.dist(self.vocab.encode(ctx))
        nd = self.hmm.next_dist(ctx[-30:])
        f = self.F @ nd
        return base, level, nd, f

    def _combine(self, base, f, use_hmm=True, beta=None):
        beta = self.beta if beta is None else beta
        return base * f ** beta if (use_hmm and beta > 0) else base

    def _mask(self, p, prefix):
        q = p.copy()
        q[UNK] = 0.0
        q[BOS] = 0.0
        if prefix:
            keep = np.zeros_like(q)
            keep[self.vocab.with_prefix(prefix)] = 1.0
            q *= keep
        s = q.sum()
        return q / s if s > 0 else q

    def topk_words(self, ctx, prefix="", k=3, use_hmm=True):
        base, _, _, f = self._parts(ctx)
        q = self._mask(self._combine(base, f, use_hmm), prefix)
        if q.sum() <= 0:
            return []
        kk = min(k, len(q) - 1)
        idx = np.argpartition(-q, kk)[:kk]
        idx = idx[np.argsort(-q[idx])]
        return [self.vocab.itos[i] for i in idx if q[i] > 0]

    # ---- public API
    def suggest(self, text, k=3, user=None, use_hmm=True):
        t0 = time.perf_counter()
        ctx, prefix = parse(text)
        ctx = ctx[-40:]
        base, level, nd, f = self._parts(ctx)
        final = self._mask(self._combine(base, f, use_hmm), prefix)
        plain = self._mask(base, prefix)
        extra, boosted = [], set()
        if user is not None:
            final, extra, boosted = user.blend(final, ctx, prefix, self.vocab)
        cands = []
        if final.sum() > 0:
            kk = min(k, len(final) - 1)
            for i in np.argpartition(-final, kk)[:kk]:
                if final[i] > 0:
                    cands.append((self.vocab.itos[i], float(final[i]), int(i)))
        cands += [(w, float(x), None) for w, x in extra]
        cands.sort(key=lambda c: -c[1])
        T = self.hmm.T
        out = []
        for w, pr, i in cands[:k]:
            eos = i == EOS
            pos = "END" if eos else self.hmm.tags[int(np.argmax(self.hmm.ratio(w) * nd[:T]))]
            out.append({"word": w, "prob": round(pr, 4),
                        "base_prob": round(float(plain[i]), 4) if i is not None else None,
                        "pos": pos, "eos": eos, "personal": w in boosted})
        words = ctx[-20:]
        tags = self.hmm.viterbi(words)
        names = self.hmm.tags + ["END"]
        order = np.argsort(-nd)[:4]
        return {
            "mode": "completion" if prefix else "next_word",
            "prefix": prefix,
            "context": ctx[-6:],
            "order": LEVELS[level],
            "oov": [w for w in ctx if w not in self.vocab.stoi],
            "suggestions": out,
            "tags": [{"word": w, "tag": t} for w, t in zip(words, tags)],
            "next_tags": [{"tag": names[i], "p": round(float(nd[i]), 4)} for i in order],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def complete(self, text, length=4, beam=3, use_hmm=True):
        """Beam-search phrase completion (the first word respects the prefix being typed)."""
        ctx, prefix = parse(text)
        ctx = ctx[-40:]
        beams = [(0.0, [], False)]
        for step in range(length):
            nxt = []
            for score, ws, done in beams:
                if done:
                    nxt.append((score, ws, True))
                    continue
                base, _, _, f = self._parts(ctx + ws)
                q = self._mask(self._combine(base, f, use_hmm), prefix if step == 0 else "")
                if q.sum() <= 0:
                    nxt.append((score, ws, True))
                    continue
                for i in np.argpartition(-q, beam)[:beam]:
                    if q[i] <= 0:
                        continue
                    ended = int(i) == EOS
                    nxt.append((score + math.log(q[i]),
                                ws if ended else ws + [self.vocab.itos[i]], ended))
            beams = sorted(nxt, key=lambda b: -b[0])[:beam]
        out, seen = [], set()
        for score, ws, done in sorted(beams, key=lambda b: -b[0] / (len(b[1]) + 1)):
            t = " ".join(ws)
            if t and t not in seen:
                seen.add(t)
                out.append({"text": t, "ended": done, "score": round(score / (len(ws) + 1), 3)})
        return out
