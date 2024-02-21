import os
from . import job_io


def checkpoint(job, logger, func, cache_path):
    if os.path.exists(cache_path) and job["cache"]:
        logger.info("{:s}, read cache".format(func.__name__))
        return job_io.read(path=cache_path)
    else:
        logger.info("{:s}, run".format(func.__name__))

        job = func(job=job, logger=logger)

        if job["cache"]:
            logger.info("{:s}, write cache".format(func.__name__))
            job_io.write(path=cache_path, job=job)

    return job
