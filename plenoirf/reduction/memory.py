def make_config_if_None(memory_config=None):
    if memory_config is None:
        return {"use_tmp_dir": False, "read_buffer_size": 0}
    else:
        return memory_config


def make_config_for_hpc_nfs():
    return {"use_tmp_dir": True, "read_buffer_size": "1G"}


def make_config(scheme):
    if scheme == "hpc-nfs":
        return make_config_for_hpc_nfs()
    else:
        return make_config_if_None(None)
