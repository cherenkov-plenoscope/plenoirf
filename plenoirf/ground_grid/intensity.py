import numpy as np
import gzip
import os
import zipfile

from .. import bookkeeping


class Reader:
    def __init__(
        self,
        path,
        record_dtype=[("x_bin", "i4"), ("y_bin", "i4"), ("size", "f8")],
    ):
        self.path = path
        self.zip = zipfile.ZipFile(self.path, "r")
        self.uids = []
        self.record_dtype = record_dtype
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
