import rename_after_writing as rnw
import os

from .. import reduction


class Register:
    def __init__(self, map_dir):
        self.lock = False
        self.map_dir = map_dir
        self._register_path = os.path.join(self.map_dir, _filename())
        self._locked_register_path = self._register_path + ".lock"

        if os.path.exists(self._locked_register_path):
            raise RuntimeError(
                f"Can not aquire lock on '{self._register_path:s}'."
            )
        else:
            if not os.path.exists(self._register_path):
                _write_register(path=self._register_path, run_ids=[])

        rnw.move(src=self._register_path, dst=self._locked_register_path)
        self.lock = True

    def __enter__(self):
        return self

    def add_run_ids(self, run_ids):
        assert self.lock
        old_register = _read_register(path=self._locked_register_path)
        new_register = set.union(
            set(old_register),
            set(run_ids),
        )
        new_register = list(new_register)
        new_register = sorted(new_register)
        _write_register(path=self._locked_register_path, run_ids=new_register)

    def reset(self):
        assert self.lock
        run_ids = reduction.list_run_ids_ready_for_reduction(
            map_dir=self.map_dir
        )
        _write_register(path=self._locked_register_path, run_ids=run_ids)

    def get_run_ids(self):
        assert self.lock
        run_ids_ready_for_reduction = (
            reduction.list_run_ids_ready_for_reduction(map_dir=self.map_dir)
        )
        run_ids_in_register = _read_register(path=self._locked_register_path)

        run_ids = set.union(
            set(run_ids_in_register),
            set(run_ids_ready_for_reduction),
        )
        run_ids = list(run_ids)
        return sorted(run_ids)

    def close(self):
        assert self.lock
        rnw.move(src=self._locked_register_path, dst=self._register_path)
        self.lock = False

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def __repr__(self):
        return f"{self.__class__.__name__:s}(map_dir='{self.map_dir:s}')"


def _write_register(path, run_ids):
    sorted_run_ids = sorted(run_ids)
    with rnw.open(path, "wt") as f:
        for run_id in sorted_run_ids:
            f.write(f"{run_id:06d}\n")


def _read_register(path):
    run_ids = []
    with open(path, "rt") as f:
        for line in f.readlines():
            run_ids.append(int(line))
    return sorted(run_ids)


def _filename():
    return "run_id_register.txt"
