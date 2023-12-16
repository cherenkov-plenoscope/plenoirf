import corsika_primary as cpw
import numpy as np


def mask_cherenkov_emission(corsika_particles, corsika_particle_zoo):
    num_particles = corsika_particles.shape[0]
    media = corsika_particle_zoo.media_cherenkov_threshold_lorentz_factor
    out = {}

    out["unknown"] = np.zeros(num_particles, dtype=np.int)
    out["media"] = {}

    for medium_key in media:
        out["media"][medium_key] = np.zeros(num_particles, dtype=np.int)

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
