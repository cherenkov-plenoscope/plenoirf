import numpy as np
import gzip
import os
import io
import tarfile
import zipfile
import shutil
from .. import bookkeeping


def histogram_to_bytes(img):
    img_f4 = img.astype("<f4")
    img_f4_flat_c = img_f4.flatten(order="c")
    img_f4_flat_c_bytes = img_f4_flat_c.tobytes()
    img_gzip_bytes = gzip.compress(img_f4_flat_c_bytes)
    return img_gzip_bytes


def bytes_to_histogram(img_bytes_gz):
    img_bytes = gzip.decompress(img_bytes_gz)
    arr = np.frombuffer(img_bytes, dtype="<f4")
    num_bins = arr.shape[0]
    num_bins_edge = int(np.sqrt(num_bins))
    assert num_bins_edge * num_bins_edge == num_bins
    return arr.reshape((num_bins_edge, num_bins_edge), order="c")


# histograms
# ----------
# A dict with the unique-id (uid) as key for the airshowers, containing the
# gzip-bytes to be read with bytes_to_histogram()


def read_all_histograms(path):
    grids = {}
    with tarfile.open(path, "r") as tarfin:
        for tarinfo in tarfin:
            idx = int(tarinfo.name[0 : bookkeeping.uid.UID_NUM_DIGITS])
            grids[idx] = tarfin.extractfile(tarinfo).read()
    return grids


def read_histograms(path, indices=None):
    if indices is None:
        return read_all_histograms(path)
    else:
        indices_set = set(indices)
        grids = {}
        with tarfile.open(path, "r") as tarfin:
            for tarinfo in tarfin:
                idx = int(tarinfo.name[0 : bookkeeping.uid.UID_NUM_DIGITS])
                if idx in indices_set:
                    grids[idx] = tarfin.extractfile(tarinfo).read()
        return grids


def write_histograms(path, grid_histograms):
    with tarfile.open(path + ".tmp", "w") as tarfout:
        for idx in grid_histograms:
            filename = bookkeeping.uid.UID_FOTMAT_STR.format(idx) + ".f4.gz"
            with io.BytesIO() as buff:
                info = tarfile.TarInfo(filename)
                info.size = buff.write(grid_histograms[idx])
                buff.seek(0)
                tarfout.addfile(info, buff)
    shutil.move(path + ".tmp", path)


def reduce(list_of_grid_paths, out_path):
    with tarfile.open(out_path + ".tmp", "w") as tarfout:
        for grid_path in list_of_grid_paths:
            with tarfile.open(grid_path, "r") as tarfin:
                for tarinfo in tarfin:
                    tarfout.addfile(
                        tarinfo=tarinfo, fileobj=tarfin.extractfile(tarinfo)
                    )
    shutil.move(out_path + ".tmp", out_path)


class GridReader:
    def __init__(self, path):
        self.path = str(path)
        self.tar = tarfile.open(name=self.path, mode="r|")
        self.next_info = self.tar.next()

    def __next__(self):
        if self.next_info is None:
            raise StopIteration

        idx = int(self.next_info.name[0 : bookkeeping.uid.UID_NUM_DIGITS])
        bimg = self.tar.extractfile(self.next_info).read()
        img = bytes_to_histogram(bimg)
        self.next_info = self.tar.next()
        return idx, img

    def close(self):
        self.tar.close()

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        out = "{:s}(path='{:s}')".format(self.__class__.__name__, self.path)
        return out
