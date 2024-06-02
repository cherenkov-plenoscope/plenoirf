import numpy as np
import tempfile
import time
import os


def disk_writing(path, seed=1):
    out = {}
    with tempfile.TemporaryDirectory(
        suffix="-benchmark",
        prefix="plenoirf-",
        dir=path,
        ignore_cleanup_errors=False,
    ) as td:
        # 1k
        dts = []
        for i in range(27):
            dt = benchmark_open_and_write(
                path=os.path.join(path, "{:06d}.rnd".format(i)),
                seed=i,
                num_blocks=1,
                block_size=1000,
            )
            dts.append(dt)
        out["1e3"] = _analysis(dts=dts, size=1e6)

        # 1M
        dts = []
        for i in range(9):
            dt = benchmark_open_and_write(
                path=os.path.join(path, "{:06d}.rnd".format(i)),
                seed=i,
                num_blocks=1,
                block_size=1000 * 1000,
            )
            dts.append(dt)
        out["1e6"] = _analysis(dts=dts, size=1e6)

        # 1G
        dts = []
        for i in range(3):
            dt = benchmark_open_and_write(
                path=os.path.join(path, "{:06d}.rnd".format(i)),
                seed=i,
                num_blocks=1000,
                block_size=1000 * 1000,
            )
            dts.append(dt)
        out["1e9"] = _analysis(dts=dts, size=1e9)

    return out


def _analysis(dts, size):
    dts = np.array(dts)
    rates_MB_per_s = (1e-6 * size) / dts
    out = {}
    out["rate_MB_per_s"] = {}
    out["rate_MB_per_s"]["avg"] = np.average(rates_MB_per_s)
    out["rate_MB_per_s"]["std"] = np.std(rates_MB_per_s)
    return out


def benchmark_open_and_write(path, seed, num_blocks=1, block_size=1000):
    prng = np.random.Generator(np.random.PCG64(1))
    start = time.time()
    with open(path, "wb") as fout:
        for i in range(num_blocks):
            block = prng.bytes(block_size)
            fout.write(block)

    stop = time.time()
    return stop - start
