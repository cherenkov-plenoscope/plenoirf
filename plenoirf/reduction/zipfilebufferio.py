import io
import gzip
import sparse_numeric_table as snt
import zipfile


class ZipFileBufferIO:
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        self.zin = zipfile.ZipFile(file=self.file, mode="r")
        return self

    @property
    def filenames(self):
        return [i.filename for i in self.zin.filelist]

    def read(self, path, mode):
        with self.zin.open(path) as fin:
            if "|gz" in mode:
                block = gzip.decompress(fin.read())
            else:
                block = fin.read()
        if "t" in mode:
            buff = io.StringIO()
            buff.write(bytes.decode(block))
        elif "b" in mode:
            buff = io.BytesIO()
            buff.write(block)
        else:
            raise KeyError("mode must either be 'b' or 't'.")
        buff.seek(0)
        return buff

    def close(self):
        return self.zin.close()

    def read_event_table(self, path):
        with snt.open(file=self.read(path, "rb"), mode="r") as part:
            return part.query()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"
