import plenoirf
import tempfile
import os
import gzip
import pytest


def test_gzip_file_roundtrip():
    for block_size in [2**23, 2**1]:
        with tempfile.TemporaryDirectory(prefix="plenoirf-") as tmp_dir:
            content = "abc die Katze lief im Schnee."
            path = os.path.join(tmp_dir, "one.txt")
            with open(path, "wt") as f:
                f.write(content)

            plenoirf.utils.gzip_file(path, block_size=block_size)

            assert os.path.exists(path + ".gz")
            assert not os.path.exists(path)
            with open(path + ".gz", "rb") as f:
                back = gzip.decompress(f.read()).decode()
            assert content == back

            plenoirf.utils.gunzip_file(path + ".gz", block_size=block_size)

            assert not os.path.exists(path + ".gz")
            assert os.path.exists(path)
            with open(path, "rt") as f:
                back2 = f.read()
            assert content == back2
