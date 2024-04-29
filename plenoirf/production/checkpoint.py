import os
import pickle
import rename_after_writing as rnw


def checkpoint(job, logger, func, cache_path, block_id=None, blk=None):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path) and job["cache"]:
        logger.info("{:s}, read cache".format(func.__name__))
        with open(cache_path, "rb") as fin:
            return pickle.loads(fin.read())
    else:
        logger.info("{:s}, run".format(func.__name__))

        if block_id is None:
            job = func(job=job, logger=logger)
        else:
            job = func(job=job, logger=logger, block_id=block_id, blk=blk)

        if job["cache"]:
            logger.info("{:s}, write cache".format(func.__name__))
            with rnw.open(cache_path, "wb") as fout:
                fout.write(pickle.dumps(job))

    return job
