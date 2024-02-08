from plenoirf.production import cherenkov_bunch_storage
import corsika_primary as cpw
import numpy as np
import tempfile
import os


def test_mask():
    SIZE = 123456
    bunches = np.zeros(shape=(SIZE, cpw.I.BUNCH.NUM_FLOAT32), dtype=np.float32)

    for col in range(bunches.shape[1]):
        bunches[:, col] = np.arange(SIZE)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bunches.tar")

        with cpw.cherenkov.CherenkovEventTapeWriter(
            path=path, buffer_capacity=24
        ) as f:
            f.write_runh(cherenkov_bunch_storage._make_fake_runh())
            for event_number in [1]:
                f.write_evth(cherenkov_bunch_storage._make_fake_evth())
                f.write_payload(bunches)

        prng = np.random.Generator(np.random.PCG64(14))
        SUBSIZE = 1000
        choice = prng.choice(SIZE, replace=False, size=SUBSIZE)

        assert len(set(choice)) == SUBSIZE
        assert len(choice) == SUBSIZE

        subbunches = cherenkov_bunch_storage.read_with_mask(
            path=path,
            bunch_indices=choice,
        )

        assert subbunches.shape[0] == SUBSIZE

        choice = sorted(choice)

        for ii, cc in enumerate(choice):
            np.testing.assert_almost_equal(subbunches[ii, 0], cc)
