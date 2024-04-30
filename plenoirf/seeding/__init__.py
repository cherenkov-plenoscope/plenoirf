import numpy as np
import io
import json_utils
import rename_after_writing
import json_line_logger


def make_named_random_seeds(run_id, names=[]):
    """
    Assigns random seeds to names baesd on a hash constructed from the name and
    an input seed.

    Parameters
    ----------
    run_id : int, numpy.uint64
        Must be >= 0. The input seed.
    names : list of str
        The names to receive a seed.

    Returns
    -------
    seeds : dict
        Assings names (str) to seeds (numpy.uint64)
    """
    assert len(names) == len(set(names)), "Expected names to be unique."
    assert run_id >= 0, "Expected run_id >= 0."

    out = {}
    for name in names:
        out[name] = make_seed_based_on_run_id_and_name(
            run_id=run_id,
            name=name,
        )
    return out


def write(path, named_random_seeds):
    with rename_after_writing.open(path, "wt") as fout:
        fout.write(json_utils.dumps(named_random_seeds, indent=4))


def read(path):
    with open(path, "rt") as fout:
        named_random_seeds = json_utils.loads(fout.read())
    return named_random_seeds


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


def init_numpy_random_Generator_PCG64_from_path_and_name(path, name):
    named_random_seeds = read(path=path)
    return np.random.Generator(np.random.PCG64(named_random_seeds[name]))


def make_seed_based_on_run_id_and_name(run_id, name):
    run_id = np.uint64(run_id)
    name_seed = io.BytesIO()
    name_seed.write(run_id.tobytes())
    name_seed.write(name.encode())
    name_seed.seek(0)
    return hash_PCG64(bytes=name_seed.read())


class Section:
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

    def __init__(self, run_id, module, block_id=None, logger=None):
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
        if self.block_id:
            self.name += ".block{:06d}".format(self.block_id)

        self.time_delta = json_line_logger.TimeDelta(
            logger=self.logger,
            name=self.name,
        )
        self.seed = make_seed_based_on_run_id_and_name(
            run_id=self.run_id,
            name=self.name,
        )
        self.logger.info(
            "Section: name: '{:s}', run_id: {:d}, seed: {:d}".format(
                self.name, self.run_id, self.seed
            )
        )

    def __enter__(self):
        self.time_delta.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.time_delta.__exit__()
