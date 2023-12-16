import tarfile
import io
import rename_after_writing as rnw
import uuid
import os


def tar_open_append_close(path, filename, filebytes):
    tmp_path = path + "." + uuid.uuid4().__str__()
    if os.path.exists(path):
        rnw.move(src=path, dst=tmp_path)
    with tarfile.open(name=tmp_path, mode="a") as tarout:
        filenames = [i.name for i in tarout.members]
        assert filename not in filenames, "Filename already exists."
        tar_append(tarout=tarout, filename=filename, filebytes=filebytes)
    rnw.move(src=tmp_path, dst=path)


def tar_append(tarout, filename, filebytes):
    with io.BytesIO() as buff:
        info = tarfile.TarInfo(filename)
        info.size = buff.write(filebytes)
        buff.seek(0)
        tarout.addfile(info, buff)
