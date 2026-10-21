from collections import Counter

import numpy as np

from src.ngram import UNK, EOS, BOS


class POSHMM:
    """First-order HMM over part-of-speech tags.
    - viterbi(words): most likely tag sequence
    - next_dist(words): P(next tag or END | words so far) with the forward algorithm
    Emission likelihoods are used through r(t|w) = P(t|w) / P(t), proportional to P(w|t) for a fixed word,
    so unknown words can be handled with a suffix model."""

    def __init__(self, tagged_sents, alpha=0.1):
        self.alpha = alpha
        self.tags = sorted({t for s in tagged_sents for _, t in s})
        T = self.T = len(self.tags)
        tid = {t: i for i, t in enumerate(self.tags)}
        trans = np.zeros((T + 1, T + 1))          # rows: tags + START, cols: tags + END
        tagc = np.zeros(T)
        lex = {}
        for s in tagged_sents:
            prev = T
            for w, t in s:
                j = tid[t]
                trans[prev, j] += 1
                prev = j
                tagc[j] += 1
                lex.setdefault(w, np.zeros(T))[j] += 1
            trans[prev, T] += 1
        n_tok = tagc.sum()
        self.A = (trans + 0.01) / (trans + 0.01).sum(1, keepdims=True)
        self.logA = np.log(self.A)
        self.tok_prior = tagc / n_tok
        self.end_prior = len(tagged_sents) / n_tok
        self.lex = lex
        self.suf = {1: {}, 2: {}, 3: {}}           # suffix statistics from rare words
        for w, c in lex.items():
            if c.sum() <= 5:
                for n in (1, 2, 3):
                    if len(w) > n:
                        self.suf[n].setdefault(w[-n:], np.zeros(T))
                        self.suf[n][w[-n:]] += c
        self._cache = {}

    # ---- emissions
    def posterior(self, w):
        c = self.lex.get(w)
        if c is None:
            for n in (3, 2, 1):
                s = self.suf[n].get(w[-n:]) if len(w) > n else None
                if s is not None and s.sum() >= 5:
                    c = s
                    break
        if c is None:
            return self.tok_prior
        return (c + self.alpha * self.tok_prior) / (c.sum() + self.alpha)

    def ratio(self, w):
        r = self._cache.get(w)
        if r is None:
            r = self.posterior(w) / self.tok_prior
            self._cache[w] = r
        return r

    # ---- inference
    def next_dist(self, words):
        T = self.T
        if not words:
            return self.A[T].copy()
        a = self.A[T, :T] * self.ratio(words[0])
        a /= a.sum()
        for w in words[1:]:
            a = (a @ self.A[:T, :T]) * self.ratio(w)
            a /= a.sum()
        return a @ self.A[:T]                      # length T+1 (last entry = END)

    def viterbi(self, words):
        if not words:
            return []
        T = self.T
        delta = self.logA[T, :T] + np.log(self.ratio(words[0]))
        back = []
        for w in words[1:]:
            scores = delta[:, None] + self.logA[:T, :T]
            back.append(scores.argmax(0))
            delta = scores.max(0) + np.log(self.ratio(w))
        best = int(delta.argmax())
        path = [best]
        for b in reversed(back):
            best = int(b[best])
            path.append(best)
        return [self.tags[i] for i in reversed(path)]

    def vocab_matrix(self, vocab):
        """F[w, t'] = P(t'|w)/P(t')  (END column for </s>). f = F @ next_dist is the syntactic
        compatibility of every vocabulary word with the next position."""
        F = np.ones((vocab.V, self.T + 1))
        for i, w in enumerate(vocab.itos):
            if i in (UNK, BOS):
                continue
            if i == EOS:
                F[i] = 0.0
                F[i, self.T] = 1.0 / self.end_prior
            else:
                F[i, :self.T] = self.ratio(w)
                F[i, self.T] = 0.0
        return F

    def evaluate(self, tagged_sents):
        c = Counter()
        default = self.tags[int(np.argmax(self.tok_prior))]
        for s in tagged_sents:
            pred = self.viterbi([w for w, _ in s])
            for (w, gold), p in zip(s, pred):
                known = w in self.lex
                mft = self.tags[int(np.argmax(self.lex[w]))] if known else default
                c["n"] += 1
                c["ok"] += p == gold
                c["base"] += mft == gold
                c["kn"] += known
                c["kok"] += known and p == gold
                c["un"] += not known
                c["uok"] += (not known) and p == gold
        return {"accuracy": c["ok"] / c["n"], "baseline": c["base"] / c["n"],
                "accuracy_known": c["kok"] / max(1, c["kn"]),
                "accuracy_unknown": c["uok"] / max(1, c["un"]),
                "unknown_share": c["un"] / c["n"], "n_tokens": c["n"]}
