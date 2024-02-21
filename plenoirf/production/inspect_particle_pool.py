import corsika_primary as cpw
import numpy as np
import os
import sparse_numeric_table as spt

from .. import bookkeeping


def run_job(job, logger):
    aperture_radius_m = guess_aperture_radius_m(job=job)

    logger.info("inspect_particle_pool, init corsika particle zoo")
    corsika_particle_zoo = cpw.particles.identification.Zoo(
        media_refractive_indices={
            "water": 1.33,
            "air": cpw.particles.identification.refractive_index_atmosphere(
                altitude_asl_m=job["site"]["observation_level_asl_m"]
            ),
        }
    )

    logger.info("inspect_particle_pool, read corsika's particle output")

    with cpw.particles.ParticleEventTapeReader(
        path=os.path.join(
            job["paths"]["work_dir"],
            "simulate_shower_and_collect_cherenkov_light_in_grid",
            "particle_pools.tar.gz",
        )
    ) as particle_run:
        for event in particle_run:
            corsika_evth, particle_reader = event

            uid = make_uid_from_evth(corsika_evth=corsika_evth)

            ppp = init_particlepool_record(uid=uid)

            core = record_by_uid(
                dynamicsizerecarray=job["event_table"]["core"],
                uid=uid,
            )

            if core:
                aaa = init_particlepoolonaperture_record(uid=uid)

            for particle_block in particle_reader:
                cherenkov_emission_block_mask = mask_cherenkov_emission(
                    corsika_particles=particle_block,
                    corsika_particle_zoo=corsika_particle_zoo,
                )

                ppp = update_particlepool_record(
                    rec=ppp,
                    cherenkov_emission_mask=cherenkov_emission_block_mask,
                )

                if core:
                    """
                    If the core was drawn by the ground grid algorithm,
                    we want to know if particles emit Cherenkov light right
                    in front of the instrument's aperture.
                    """
                    particles_emitting_in_air = particle_block[
                        cherenkov_emission_block_mask["media"]["air"]
                    ]

                    aaa = update_particlepoolonaperture_record(
                        rec=aaa,
                        particles_emitting_in_air=particles_emitting_in_air,
                        aperture_x_m=core["core_x_m"],
                        aperture_y_m=core["core_y_m"],
                        aperture_radius_m=aperture_radius_m,
                    )

            job["event_table"]["particlepool"].append_record(ppp)
            if core:
                job["event_table"]["particlepoolonaperture"].append_record(aaa)
    return job


def guess_aperture_radius_m(job):
    grid_bin_width = job["config"]["ground_grid"]["geometry"]["bin_width_m"]
    grid_bin_diagonal = np.sqrt(3) * grid_bin_width
    aperture_radius_m = (1 / 2) * grid_bin_diagonal
    return aperture_radius_m


def init_particlepool_record(uid):
    ppp = {spt.IDX: uid}
    ppp["num_water_cherenkov"] = 0
    ppp["num_air_cherenkov"] = 0
    ppp["num_unknown"] = 0
    return ppp


def init_particlepoolonaperture_record(uid):
    aaa = {spt.IDX: uid}
    aaa["num_air_cherenkov_on_aperture"] = 0
    return aaa


def update_particlepool_record(rec, cherenkov_emission_mask):
    cer = cherenkov_emission_mask
    rec["num_water_cherenkov"] += np.sum(cer["media"]["water"])
    rec["num_air_cherenkov"] += np.sum(cer["media"]["air"])
    rec["num_unknown"] += np.sum(cer["unknown"])
    return rec


def update_particlepoolonaperture_record(
    rec,
    particles_emitting_in_air,
    aperture_x_m,
    aperture_y_m,
    aperture_radius_m,
):
    distance_m = distances_to_point_on_observation_level_m(
        corsika_particles=particles_emitting_in_air,
        x_m=aperture_x_m,
        y_m=aperture_y_m,
    )
    mask_on_aperture = distance_m <= aperture_radius_m
    rec["num_air_cherenkov_on_aperture"] += np.sum(mask_on_aperture)
    return rec


def make_uid_from_evth(corsika_evth):
    run_id = int(corsika_evth[cpw.I.EVTH.RUN_NUMBER])
    event_id = int(corsika_evth[cpw.I.EVTH.EVENT_NUMBER])
    uid = bookkeeping.uid.make_uid(run_id=run_id, event_id=event_id)
    return uid


def record_by_uid(dynamicsizerecarray, uid):
    for i in range(len(dynamicsizerecarray)):
        if dynamicsizerecarray._recarray[spt.IDX][i] == uid:
            out = {}
            for name in dynamicsizerecarray._recarray.dtype.names:
                out[name] = dynamicsizerecarray._recarray[name][i]
            return out
    return None


def mask_cherenkov_emission(corsika_particles, corsika_particle_zoo):
    """
    Returns binary masks indicating if a particle will emit Cherenkov light
    in a certain medium.

    Parameters
    ----------
    corsika_particles : numpy.ndarray, shape=(N, 7), dtype=np.float32
        Block of 'N' particles written by corsika.
        See corsika_primary.particles.
    corsika_particle_zoo : corsika_primary.particles.identification.Zoo
        The zoo does not only identify particles, it is also primed with
        media e.g. water and air for which it stores the threshold
        velocities the particles must pass to emit Cherenkov light.

    Returns
    -------
    masks : dict of binary masks
        -unknown : 1 if the zoo does not know the particle type.
        -media
            - e.g. water: 1 if the particle will emit Cherenkov light in water.
            - e.g. air ...
            - ...
    """
    num_particles = corsika_particles.shape[0]
    media = corsika_particle_zoo.media_cherenkov_threshold_lorentz_factor
    out = {}

    out["unknown"] = np.zeros(num_particles, dtype=bool)
    out["media"] = {}

    for medium_key in media:
        out["media"][medium_key] = np.zeros(num_particles, dtype=bool)

    for i in range(num_particles):
        particle = corsika_particles[i]

        corsika_particle_id = cpw.particles.decode_particle_id(
            code=particle[cpw.I.PARTICLE.CODE]
        )

        if corsika_particle_zoo.has(corsika_id=corsika_particle_id):
            momentum_GeV = np.array(
                [
                    particle[cpw.I.PARTICLE.PX],
                    particle[cpw.I.PARTICLE.PY],
                    particle[cpw.I.PARTICLE.PZ],
                ]
            )

            for medium_key in media:
                if corsika_particle_zoo.cherenkov_emission(
                    corsika_id=corsika_particle_id,
                    momentum_GeV=momentum_GeV,
                    medium_key=medium_key,
                ):
                    out["media"][medium_key][i] = 1
        else:
            out["unknown"][i] = 1

    return out


def distances_to_point_on_observation_level_m(corsika_particles, x_m, y_m):
    num_particles = corsika_particles.shape[0]
    distances = np.zeros(num_particles)
    for i in range(num_particles):
        par = corsika_particles[i]
        par_x_m = par[cpw.I.PARTICLE.X] * cpw.CM2M
        par_y_m = par[cpw.I.PARTICLE.Y] * cpw.CM2M
        dx = par_x_m - x_m
        dy = par_y_m - y_m
        distances[i] = np.hypot(dx, dy)
    return distances
