import numpy as np
import gzip
import os
import zipfile

from .. import bookkeeping


def make_dtype():
    return [("x_bin", "i4"), ("y_bin", "i4"), ("weight_photons", "f8")]


def assert_bins_in_limits(hist, num_bins_each_axis):
    num = num_bins_each_axis
    if any(hist["x_bin"] < 0) or any(hist["x_bin"] >= num):
        raise AssertionError(
            "merlict_c89 ground_grid_main hist x_bin out of range"
        )
    if any(hist["y_bin"] < 0) or any(hist["y_bin"] >= num):
        raise AssertionError(
            "merlict_c89 ground_grid_main hist y_bin out of range"
        )


def assert_bins_unique(hist):
    counts = {}
    for cell in hist:
        xy = (cell["x_bin"], cell["y_bin"])
        if xy in counts:
            counts[xy] += 1
        else:
            counts[xy] = 1
    for cell in counts:
        assert counts[cell] == 1, (
            "Expected bins in sparse histogram to be unique, "
            f"but bin({cell[0]:d}, {cell[1]:d}) occurs {counts[cell]:d} times."
        )


class Reader:
    def __init__(
        self,
        path,
        record_dtype=None,
    ):
        self.path = path
        self.zip = zipfile.ZipFile(self.path, "r")
        self.uids = []
        if record_dtype is None:
            self.record_dtype = make_dtype()
        for zipitem in self.zip.infolist():
            if zipitem.filename.endswith(".i4_i4_f8.gz"):
                run_id = int(os.path.dirname(zipitem.filename))
                event_id = int(os.path.basename(zipitem.filename)[0:6])
                self.uids.append(
                    bookkeeping.uid.make_uid(run_id=run_id, event_id=event_id)
                )

    def close(self):
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _read_by_filename(self, filename):
        with self.zip.open(filename) as f:
            payload_gz = f.read()
        payload = gzip.decompress(payload_gz)
        return np.fromstring(payload, dtype=self.record_dtype)

    def _read_by_uid(self, uid):
        run_id, event_id = bookkeeping.uid.split_uid(uid)
        return self._read_by_filename(
            filename=f"{run_id:06d}/{event_id:06d}.i4_i4_f8.gz"
        )

    def __getitem__(self, uid):
        return self._read_by_uid(uid=uid)

    def __iter__(self):
        return iter(self.uids)

    def __repr__(self):
        return f"{self.__class__.__name__:s}(path='{self.path:s}')"
