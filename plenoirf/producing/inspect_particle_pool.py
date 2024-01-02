import corsika_primary as cpw
import numpy as np
import sparse_numeric_table as spt

from .. import analysis
from .. import bookkeeping


def run_job(job, logger):
    bin_radius_m = job["config"]["ground_grid"]["geometry"]["bin_width_m"] / 2

    corsika_particle_zoo = cpw.particles.identification.Zoo(
        media_refractive_indices={
            "water": 1.33,
            "air": cpw.particles.identification.refractive_index_atmosphere(
                altitude_asl_m=job["site"]["observation_level_asl_m"]
            ),
        }
    )

    with cpw.particles.ParticleEventTapeReader(
        path=job["paths"]["tmp"]["particle_pools_tar"]
    ) as parrun:
        for event_idx, event in enumerate(parrun):
            corsika_evth, parreader = event

            uid = nail_down_uid(
                corsika_evth=corsika_evth, job=job, event_idx=event_idx
            )

            core = record_by_uid(
                dynamicsizerecarray=job["event_table"]["core"],
                uid=uid,
            )

            ppp = {spt.IDX: uid}
            ppp["num_water_cherenkov"] = 0
            ppp["num_air_cherenkov"] = 0
            ppp["num_unknown"] = 0

            if core:
                aaa = {spt.IDX: uid}
                aaa["num_air_cherenkov_on_aperture"] = 0

            for parblock in parreader:
                cer = analysis.particles_on_ground.mask_cherenkov_emission(
                    corsika_particles=parblock,
                    corsika_particle_zoo=corsika_particle_zoo,
                )
                ppp["num_water_cherenkov"] += np.sum(cer["media"]["water"])
                ppp["num_air_cherenkov"] += np.sum(cer["media"]["air"])
                ppp["num_unknown"] += np.sum(cer["unknown"])

                if core:
                    subparblock = parblock[cer["media"]["air"]]
                    subparblock_r_m = analysis.particles_on_ground.distances_to_point_on_observation_level_m(
                        corsika_particles=subparblock,
                        x_m=core["core_x_m"],
                        y_m=core["core_y_m"],
                    )
                    mask_on_aperture = subparblock_r_m <= bin_radius_m
                    aaa["num_air_cherenkov_on_aperture"] += np.sum(
                        mask_on_aperture
                    )

            job["event_table"]["particlepool"].append_record(ppp)
            if core:
                job["event_table"]["particlepoolonaperture"].append_record(aaa)
    return job


def nail_down_uid(corsika_evth, job, event_idx):
    run_id = int(corsika_evth[cpw.I.EVTH.RUN_NUMBER])
    assert run_id == job["run_id"]
    event_id = event_idx + 1
    assert event_id == corsika_evth[cpw.I.EVTH.EVENT_NUMBER]
    uid = bookkeeping.uid.make_uid(run_id=run_id, event_id=event_id)
    return uid


def record_by_uid(dynamicsizerecarray, uid):
    for i in range(len(dynamicsizerecarray)):
        if dynamicsizerecarray._recarray[spt.IDX][i] == uid:
            out = {}
            for name in dynamicsizerecarray._recarray.dtype.names:
                out[name] = dynamicsizerecarray._recarray[name][i]
            return out
