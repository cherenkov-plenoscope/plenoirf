import zipfile
import os


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
