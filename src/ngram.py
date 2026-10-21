import math
from bisect import bisect_left

import numpy as np

UNK, EOS, BOS = 0, 1, 2


class Vocab:
    def __init__(self, words):
        self.itos = ["<unk>", "</s>", "<s>"] + list(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.V = len(self.itos)
        real = sorted((w, i) for i, w in enumerate(self.itos) if i > 2)
        self._sorted_words = [w for w, _ in real]
        self._sorted_ids = np.array([i for _, i in real], dtype=np.int64)

    def encode(self, tokens):
        g = self.stoi.get
        return [g(t, UNK) for t in tokens]

    def with_prefix(self, prefix):
        lo = bisect_left(self._sorted_words, prefix)
        hi = bisect_left(self._sorted_words, prefix + "\uffff")
        return self._sorted_ids[lo:hi]


def _discount(counts):
    n1, n2 = int((counts == 1).sum()), int((counts == 2).sum())
    return min(0.95, n1 / (n1 + 2 * n2)) if n1 else 0.75


class KneserNey:
    """Interpolated Kneser-Ney trigram. dist(ctx) returns the FULL next-word distribution (numpy vector).
    Highest order uses raw counts, lower orders use continuation counts."""

    def __init__(self, sentences, vocab, min_tri=1):
        V = self.V = vocab.V
        self.vocab = vocab
        a, b, c = [], [], []
        for s in sentences:
            ids = [BOS, BOS] + vocab.encode(s) + [EOS]
            a += ids[:-2]; b += ids[1:-1]; c += ids[2:]
        a, b, c = (np.asarray(x, dtype=np.int64) for x in (a, b, c))
        keys, cnt = np.unique((a * V + b) * V + c, return_counts=True)
        ua, ub, uc = keys // (V * V), (keys // V) % V, keys % V
        self.D3 = _discount(cnt)

        # level 2: N1+(. v w) from ALL trigram types
        bk, bcnt = np.unique(ub * V + uc, return_counts=True)
        bv, bw = bk // V, bk % V
        self.D2 = _discount(bcnt)
        self.bi_w, self.bi_c = bw, bcnt.astype(float)
        self.bi_index = self._index(bv, self.bi_c)

        # level 1: N1+(. w), interpolated with a uniform distribution
        cont1 = np.bincount(bw, minlength=V).astype(float)
        self.D1 = _discount(cont1[cont1 > 0])
        T, U = cont1.sum(), (cont1 > 0).sum()
        valid = np.ones(V, bool); valid[BOS] = False
        p1 = np.maximum(cont1 - self.D1, 0) / T
        p1[valid] += self.D1 * U / T / valid.sum()
        self.p1 = p1

        # level 3: raw counts, pruned
        keep = cnt >= min_tri
        ua, ub, uc, cnt = ua[keep], ub[keep], uc[keep], cnt[keep].astype(float)
        self.tri_w, self.tri_c = uc, cnt
        self.tri_index = self._index(ua * V + ub, cnt)

    @staticmethod
    def _index(ctx_sorted, cnt):
        uniq, starts = np.unique(ctx_sorted, return_index=True)
        ends = np.append(starts[1:], len(ctx_sorted))
        totals = np.add.reduceat(cnt, starts)
        return {int(k): (int(s), int(e), float(t), int(e - s))
                for k, s, e, t in zip(uniq, starts, ends, totals)}

    def dist(self, ctx):
        """ctx: list of word ids (only the last two are used). Returns (probabilities, order used)."""
        v = ctx[-1] if len(ctx) >= 1 else BOS
        u = ctx[-2] if len(ctx) >= 2 else BOS
        p, level = self.p1, 1
        e = self.bi_index.get(v)
        if e:
            s, t, tot, types = e
            q = p * (self.D2 * types / tot)
            q[self.bi_w[s:t]] += (self.bi_c[s:t] - self.D2) / tot
            p, level = q, 2
        e = self.tri_index.get(u * self.V + v)
        if e:
            s, t, tot, types = e
            q = p * (self.D3 * types / tot)
            q[self.tri_w[s:t]] += (self.tri_c[s:t] - self.D3) / tot
            p, level = q, 3
        return p, level


class AddK:
    """Plain add-k n-gram (baseline). prob(ctx_ids, w) = (c(ctx w)+k) / (c(ctx)+k V)."""

    def __init__(self, sentences, vocab, n=3, k=0.1):
        V = self.V = vocab.V
        self.n, self.k = n, k
        cols = [[] for _ in range(n)]
        for s in sentences:
            ids = [BOS] * (n - 1) + vocab.encode(s) + [EOS]
            L = len(ids)
            for j in range(n):
                cols[j] += ids[j:L - (n - 1 - j)]
        full = np.zeros(len(cols[0]), dtype=np.int64)
        for col in cols:
            full = full * V + np.asarray(col, dtype=np.int64)
        kf, cf = np.unique(full, return_counts=True)
        kc, cc = np.unique(full // V, return_counts=True)
        self.full = dict(zip(kf.tolist(), cf.tolist()))
        self.ctx = dict(zip(kc.tolist(), cc.tolist()))

    def prob(self, ctx, w):
        ck = 0
        for x in ([BOS] * (self.n - 1) + list(ctx))[len(ctx):]:
            ck = ck * self.V + x
        num = self.full.get(ck * self.V + w, 0) + self.k
        den = self.ctx.get(ck, 0) + self.k * self.V
        return num / den
