import numpy as np
import gzip
import os
import zipfile
import tarfile
import dynamicsizerecarray

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
    has_duplicates = _report_duplicate_bins(hist)
    assert not has_duplicates
    # to be sure ...
    assert _bins_unique(hist=hist)


def _report_duplicate_bins(hist):
    has_duplicates = False
    counts = _count_bin_occurances(hist=hist)
    for cell in counts:
        if len(counts[cell]) > 1:
            msg = "Expected bins in sparse histogram to be unique, "
            msg += f"but bin({cell[0]:d}, {cell[1]:d}) occurs "
            msg += f"{len(counts[cell]):d} times. ("
            for val in counts[cell]:
                msg += f"{val:e}, "
            msg += f")"
            print(msg)
            has_duplicates += 1
    return has_duplicates


def _count_bin_occurances(hist):
    counts = {}
    for cell in hist:
        xy = (cell["x_bin"], cell["y_bin"])
        if xy in counts:
            counts[xy].append(cell["weight_photons"])
        else:
            counts[xy] = [cell["weight_photons"]]
    return counts


def _bins_unique(hist):
    b = hist.tobytes()
    i8f8 = np.frombuffer(b, dtype=[("xy_bin", "i8"), make_dtype()[-1]])
    xy_bins = i8f8["xy_bin"]
    num_total = xy_bins.shape[0]
    num_unique = len(set(xy_bins))
    return num_total == num_unique


def remove_duplicate_bins_hotfix_2025_08_08(hist):
    """
    On 2025-08-08 it was discovered that former
    'plenoirf.production..ImgRoiTar_append()'
    created duplicate bin entries in the histogram. It was a bug in
    dynamicsizerecarray (0.0.8 -> 0.1.0).

    The fix is to remove all but the last found bin entry.
    It seems that the duplicates have the same values due to how numpy
    allocates arrays.
    """

    counts = _count_bin_occurances(hist)
    out = dynamicsizerecarray.DynamicSizeRecarray(dtype=make_dtype())
    for xy_bin in counts:
        last_val = counts[xy_bin][-1]
        x_bin = xy_bin[0]
        y_bin = xy_bin[1]
        out.append(
            {"x_bin": x_bin, "y_bin": y_bin, "weight_photons": last_val}
        )
    return out.to_recarray()


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


class TarReader:
    def __init__(
        self,
        path,
        record_dtype=None,
    ):
        self.path = path
        self.tar = tarfile.open(self.path, "r")
        if record_dtype is None:
            self.record_dtype = make_dtype()

    def __enter__(self):
        return self

    def __next__(self):
        tarh = self.tar.next()
        if tarh is None:
            raise StopIteration
        run_id = int(tarh.name[0:6])
        event_id = int(tarh.name[7:13])
        uid = bookkeeping.uid.make_uid(run_id=run_id, event_id=event_id)
        payload_gz = self.tar.extractfile(tarh).read()
        payload = gzip.decompress(payload_gz)
        hist = np.frombuffer(payload, dtype=self.record_dtype)
        return (uid, hist)

    def __exit__(self, type, value, traceback):
        self.close()

    def __iter__(self):
        return self

    def close(self):
        self.tar.close()

    def __repr__(self):
        return f"{self.__class__.__name__:s}(path='{self.path:s}')"
