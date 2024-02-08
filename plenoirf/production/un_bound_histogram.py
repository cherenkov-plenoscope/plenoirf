import numpy as np


class UnBoundHistogram:
    def __init__(self, bin_width):
        assert bin_width > 0
        self.bin_width = bin_width
        self.bins = {}

    def assign(self, x):
        xb = x / self.bin_width
        xb = np.floor(xb).astype(int)
        unique, counts = np.unique(xb, return_counts=True)
        bins = dict(zip(unique, counts))
        for key in bins:
            if key in self.bins:
                self.bins[key] += bins[key]
            else:
                self.bins[key] = bins[key]

    def sum(self):
        total = 0
        for key in self.bins:
            total += self.bins[key]
        return total

    def argmax(self):
        if len(self.bins) == 0:
            raise RuntimeError("No values in bins yet.")
        m = 0
        max_key = -1
        for key in self.bins:
            c = self.bins[key]
            if c > m:
                max_key = key
                m = c
        return max_key

    def modus(self):
        if len(self.bins) == 0:
            raise RuntimeError("No values in bins yet.")

        modus_key = self.argmax()
        return (modus_key + 0.5) * self.bin_width

    def quantile(self, q):
        if len(self.bins) == 0:
            raise RuntimeError("No values in bins yet.")
        assert 0 <= q <= 1.0
        total = self.sum()
        target = total * q
        sorted_keys = sorted(self.bins.keys())
        part = 0
        for key in sorted_keys:
            if part + self.bins[key] < target:
                part += self.bins[key]
            else:
                break
        missing = target - part
        assert missing <= self.bins[key]
        bin_frac = missing / self.bins[key]
        bin_center = key
        bin_quantile = bin_center + bin_frac
        quantile = bin_quantile * self.bin_width
        return quantile

    def percentile(self, p):
        return self.quantile(q=p * 1e-2)

    def to_dict(self):
        return {"bin_width": self.bin_width, "bins": self.bins}

    def __repr__(self):
        out = "{:s}(bin_width={:f})".format(
            self.__class__.__name__, self.bin_width
        )
        return out
