import zipfile
import os
from os.path import join as opj
from os.path import relpath as opr
import gzip
import rename_after_writing as rnw


def write_gz(zout, inpath, outpath):
    with zout.open(outpath, mode="w") as fout:
        with open(inpath, "rb") as fin:
            fout.write(gzip.compress(fin.read()))


def write(zout, inpath, outpath):
    with zout.open(outpath, mode="w") as fout:
        with open(inpath, "rb") as fin:
            fout.write(fin.read())


class Writer:
    def __init__(self, zout, indir, outdir):
        self.zout = zout
        self.indir = indir
        self.outdir = outdir

    def write(self, path, gz=False):
        ipath = os.path.join(self.indir, path)
        opath = os.path.join(self.outdir, path)
        if gz:
            write_gz(self.zout, ipath, opath + ".gz")
        else:
            write(self.zout, ipath, opath)


def write_dir(path, zout, base_dir=""):
    for root, dirs, files in os.walk(path):
        for file in files:
            rel_path = opr(opj(root, file), opj(path))
            zout.write(
                opj(root, file),
                opj(base_dir, rel_path),
            )


def archive_dir(path, dir_path, base_dir_path=""):
    with rnw.Path(path) as tmp_path:
        with zipfile.ZipFile(tmp_path, "w") as zout:
            write_dir(path=dir_path, zout=zout, base_dir=base_dir_path)


def extract(path, out_path):
    with zipfile.ZipFile(path, "r") as zin:
        write_dir(path=dir_path, zout=zout, base_dir=base_dir_path)
