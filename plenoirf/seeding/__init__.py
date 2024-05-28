import numpy as np
import io
import json_utils
import rename_after_writing
import json_line_logger
import xmltodict
import copy
import time


def hash_PCG64(bytes):
    """
    hashes bytes into np.uint64 using the PCG64 pseudo random number generator.

    Parameters
    ----------
    bytes : bytes
        A string of bytes which is used as a seed for the hash.

    Returns
    -------
    hash : numpy.uint64
    """
    uint8s = np.frombuffer(bytes, dtype=np.uint8)
    prng = np.random.Generator(np.random.PCG64(uint8s))
    return np.frombuffer(prng.bytes(8), dtype=np.uint64)[0]


def make_seed_based_on_run_id_and_name(run_id, name, block_id=0):
    run_id = np.uint64(run_id)
    block_id = np.uint64(block_id)
    name_seed = io.BytesIO()
    name_seed.write(run_id.tobytes())
    name_seed.write(block_id.tobytes())
    name_seed.write(name.encode())
    name_seed.seek(0)
    return hash_PCG64(bytes=name_seed.read())


class SeedSection:
    """
    A seeding section makes the random seed for a given section of a production
    run. The see is based on the run's run_id and the name of a module.
    It is expected that within the context of a Section, functions from this
    module are executed and are provided the random seed of this Section.

    Secondary, a section logs the time that it takes to run it.

    Fields
    ------
    module : module
        The module / class which's name got hashed into the seed.
    seed : numpy.uint64
        The random seed for this section.
    name : str
        The module's name plus a potential extension (block_id)
    """

    def __init__(self, run_id, module, block_id=0, logger=None):
        """
        Parameters
        ----------
        run_id : int
            The production run's run_id. The value limited due to CORSIKA.
            In a production run, CORSIKA is called once and uses the run_id
            as its seed.
        module : module / class
            A python module or class with attribute ``__name__``.
            The module's name will be hashed together with the run_id in order
            to provide the seed.
        """
        self.run_id = run_id
        self.module = module
        self.logger = json_line_logger.LoggerStdout_if_logger_is_None(
            logger=logger
        )
        self.name = self.module.__name__
        self.block_id = block_id

        self.seed = make_seed_based_on_run_id_and_name(
            run_id=self.run_id, name=self.name, block_id=self.block_id
        )

    def __enter__(self):
        self.start = time.time()
        msg = json_line_logger.xml(
            "SeedSection",
            name=self.name,
            run_id=self.run_id,
            block_id=self.block_id,
            seed=self.seed,
            status="enter",
        )
        self.logger.info(msg)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop = time.time()
        msg = json_line_logger.xml(
            "SeedSection",
            name=self.name,
            run_id=self.run_id,
            block_id=self.block_id,
            seed=self.seed,
            status="exit",
            delta=self.delta(),
        )
        self.logger.info(msg)

    @staticmethod
    def parse_log_message(log_message):
        o = xmltodict.parse(log_message)["SeedSection"]
        out = {}
        out["name"] = o["@name"]
        out["run_id"] = int(o["@run_id"])
        out["block_id"] = int(o["@block_id"])
        out["seed"] = int(o["@seed"])
        out["status"] = o["@status"]
        if out["status"] == "exit":
            out["delta"] = o["@delta"]
        return out

    @staticmethod
    def parse_json_lines_log_entry(log_entry):
        out = copy.deepcopy(log_entry)
        out["m"] = SeedSection.parse_log_message(log_message=out["m"])
        return out

    def delta(self):
        return self.stop - self.start
