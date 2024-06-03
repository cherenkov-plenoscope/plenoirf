import numpy as np
import tempfile
import time
import os
import corsika_primary
import atmospheric_cherenkov_response as acr
import spherical_coordinates
import hashlib


def disk_write_rate():
    seed = 1
    out = {}
    with tempfile.TemporaryDirectory(
        prefix="plenoirf-",
        suffix="-disk_write_rate-benchmark",
    ) as tmp:
        # 1k
        dts = []
        for i in range(27):
            dt, sz = benchmark_open_and_write(
                path=os.path.join(tmp, "{:06d}.rnd".format(i)),
                seed=i,
                num_blocks=1,
                block_size=1000,
            )
            dts.append(dt)
        out["1k"] = _analysis(dts=dts, size=sz)

        # 1M
        dts = []
        for i in range(9):
            dt, sz = benchmark_open_and_write(
                path=os.path.join(tmp, "{:06d}.rnd".format(i)),
                seed=i,
                num_blocks=1,
                block_size=1000 * 1000,
            )
            dts.append(dt)
        out["1M"] = _analysis(dts=dts, size=sz)

        # 100M
        dts = []
        for i in range(3):
            dt, sz = benchmark_open_and_write(
                path=os.path.join(tmp, "{:06d}.rnd".format(i)),
                seed=i,
                num_blocks=100,
                block_size=1000 * 1000,
            )
            dts.append(dt)
        out["100M"] = _analysis(dts=dts, size=sz)

    return out


def disk_create_write_close_open_read_remove_latency(num=1000):
    start = time.time()
    with tempfile.TemporaryDirectory(
        prefix="plenoirf-", suffix="-disk_latency-benchmark"
    ) as td:
        for i in range(num):
            s = "{:06d}".format(i)
            with open(os.path.join(td, s), "wt") as fout:
                fout.write(s)
        for i in range(num):
            s = "{:06d}".format(i)
            with open(os.path.join(td, s), "rt") as fin:
                _ = fin.read()
    stop = time.time()
    delta = (stop - start) / num
    relunc = np.sqrt(num) / num
    delta_unc = delta * relunc
    return {"avg": delta, "std": delta_unc}


def corsika():
    """
    run the same corsika run as a benchmark and measure its wall time.
    """
    opj = os.path.join
    corsika_primary_steering = make_corsika_run_steering(
        run_id=19,
        num_events=20,
    )

    event_reports = []
    with tempfile.TemporaryDirectory(
        prefix="plenoirf-",
        suffix="-corsika-benchmark",
    ) as tmp:
        md5 = hashlib.md5()
        t_run_start = time.time()
        with corsika_primary.CorsikaPrimary(
            steering_dict=corsika_primary_steering,
            stdout_path=opj(tmp, "corsika.stdout.txt"),
            stderr_path=opj(tmp, "corsika.stderr.txt"),
            particle_output_path=opj(tmp, "particle_pools.dat"),
        ) as run:
            t_run_ready = time.time()
            runh = run.runh

            md5.update(runh.tobytes())

            while True:
                try:
                    t_event_start = time.time()

                    event = next(run)
                    evth, cherenkov_reader = event
                    num_cherenkov_bunches = 0
                    md5.update(evth.tobytes())
                    for cherenkov_block in cherenkov_reader:
                        md5.update(cherenkov_block.tobytes())
                        num_cherenkov_bunches += len(cherenkov_block)

                    t_event_stop = time.time()

                    report = {
                        "run_id": evth[corsika_primary.I.EVTH.RUN_NUMBER],
                        "event_id": evth[corsika_primary.I.EVTH.EVENT_NUMBER],
                        "energy_GeV": evth[
                            corsika_primary.I.EVTH.TOTAL_ENERGY_GEV
                        ],
                        "particle_id": evth[
                            corsika_primary.I.EVTH.PARTICLE_ID
                        ],
                        "num_cherenkov_bunches": num_cherenkov_bunches,
                        "time": t_event_stop - t_event_start,
                    }
                    event_reports.append(report)
                except StopIteration as err:
                    break

        t_run_stop = time.time()

    energy_rate_GeV_per_s = []
    cherenkov_bunches_per_s = []
    for report in event_reports:
        e_rate = report["energy_GeV"] / report["time"]
        c_rate = report["num_cherenkov_bunches"] / report["time"]
        energy_rate_GeV_per_s.append(e_rate)
        cherenkov_bunches_per_s.append(c_rate)

    out = {
        "md5": md5.hexdigest(),
        "total": t_run_stop - t_run_start,
        "initializing": t_run_ready - t_run_start,
        "events": event_reports,
        "energy_rate_GeV_per_s": {
            "avg": np.mean(energy_rate_GeV_per_s),
            "std": np.std(energy_rate_GeV_per_s),
        },
        "cherenkov_bunch_rate_per_s": {
            "avg": np.mean(cherenkov_bunches_per_s),
            "std": np.std(cherenkov_bunches_per_s),
        },
    }
    return out


def _analysis(dts, size):
    dts = np.array(dts)
    rates_MB_per_s = (1e-6 * size) / dts
    out = {}
    out["rate_MB_per_s"] = {}
    out["rate_MB_per_s"]["avg"] = np.average(rates_MB_per_s)
    out["rate_MB_per_s"]["std"] = np.std(rates_MB_per_s)
    return out


def benchmark_open_and_write(path, seed, num_blocks=1, block_size=1000):
    prng = np.random.Generator(np.random.PCG64(1))
    start = time.time()
    total_size = 0
    with open(path, "wb") as fout:
        for i in range(num_blocks):
            block = prng.bytes(block_size)
            total_size += fout.write(block)

    stop = time.time()
    return stop - start, total_size


def make_corsika_run_steering(run_id=19, num_events=10):
    i8 = np.int64
    f8 = np.float64

    site = acr.sites.init("namibia")
    particle = acr.particles.init("proton")

    srun = {
        "run_id": i8(run_id),
        "event_id_of_first_event": i8(1),
        "observation_level_asl_m": f8(site["observation_level_asl_m"]),
        "earth_magnetic_field_x_muT": f8(site["earth_magnetic_field_x_muT"]),
        "earth_magnetic_field_z_muT": f8(site["earth_magnetic_field_z_muT"]),
        "atmosphere_id": i8(site["corsika_atmosphere_id"]),
        "energy_range": {
            "start_GeV": f8(particle["corsika"]["min_energy_GeV"]),
            "stop_GeV": f8(2500),
        },
        "random_seed": corsika_primary.random.seed.make_simple_seed(run_id),
    }

    prng = np.random.Generator(np.random.PCG64(run_id))
    sprimaries = []
    for ii in range(num_events):
        ene = corsika_primary.random.distributions.draw_power_law(
            prng=prng,
            lower_limit=srun["energy_range"]["start_GeV"],
            upper_limit=srun["energy_range"]["stop_GeV"],
            power_slope=-1.5,
            num_samples=1,
        )[0]

        (
            az_rad,
            zd_rad,
        ) = corsika_primary.random.distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=0.0,
            zenith_rad=0.0,
            min_scatter_opening_angle_rad=0.0,
            max_scatter_opening_angle_rad=corsika_primary.MAX_ZENITH_DISTANCE_RAD,
        )

        prm = {
            "particle_id": f8(particle["corsika"]["particle_id"]),
            "energy_GeV": f8(ene),
            "theta_rad": f8(spherical_coordinates.corsika.zd_to_theta(zd_rad)),
            "phi_rad": f8(spherical_coordinates.corsika.az_to_phi(az_rad)),
            "depth_g_per_cm2": f8(0.0),
        }
        sprimaries.append(prm)

    return {"run": srun, "primaries": sprimaries}
