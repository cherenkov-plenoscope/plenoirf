import zipfile
import tarfile
import os
import io
import glob
import sparse_numeric_table as snt
import gzip

from .. import event_table


def recude(run_paths, out_dir):
    evttab = snt.init(dtypes=event_table.structure.dtypes())
    opj = os.path.join

    for run_path in run_paths:
        run_basename = os.path.basename(run_path)
        run_id_str = os.path.splitext(run_basename)[0]
        run_id = int(run_id_str)

        with zipfile.ZipFile(run_path, "r") as zin:
            with zin.open(opj(run_id_str, "event_table.tar.gz")) as fin:
                buff = io.BytesIO()
                buff.write(gzip.decompress(fin.read()))
                buff.seek(0)
                part_evttab = snt.read(fileobj=buff, dynamic=False)

                snt.write(path=run_path + ".evttab.tar", table=part_evttab)
                evttab = snt.append(evttab, part_evttab)

    snt.write(path=opj(out_dir, "event_table.tar"), table=evttab)
