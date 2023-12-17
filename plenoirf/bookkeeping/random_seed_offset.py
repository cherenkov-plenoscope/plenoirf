def assert_in_range(random_seed_offset):
    assert 0 <= random_seed_offset < 1000


def assert_is_unique(random_seed_offsets):
    num_unique_random_seed_offsets = len(set(random_seed_offsets))
    assert num_unique_random_seed_offsets == len(random_seed_offsets)


def assert_valid_dict(obj, key="random_seed_offset"):
    for k in obj:
        assert_in_range(random_seed_offset=obj[k][key])
    assert_is_unique(random_seed_offsets=[obj[k][key] for k in obj])


def combine_into_valid_union(last, fresh, key="random_seed_offset"):
    out = {}
    names = list(set(list(fresh.keys()) + list(last.keys())))
    for name in names:
        if name in fresh and name in last:
            msg = "Then random_seed_offset of {:s} must not change!".format(
                name
            ) + "It used to be {:d}, but now it is {:d}.".format(
                int(last[name][key]), int(fresh[name][key])
            )
            assert fresh[name][key] == last[name][key], msg
            out[name] = fresh[name]
        elif name in fresh:
            out[name] = fresh[name]
        else:
            out[name] = last[name]
    return out
