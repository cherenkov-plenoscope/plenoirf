#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import glob
import json_utils
import corsika_primary

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)
passing_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
zoo = corsika_primary.particles.identification.Zoo(
    media_refractive_indices={"water": 1.33}
)


radius_m = 1e4

RRR = {}

for pk in ["proton"]:  # PARTICLES:

    uid_common = snt.logic.intersection(
        passing_trigger[pk]["uid"],
        passing_quality[pk]["uid"],
    )

    event_table = res.event_table(particle_key=pk).query(
        levels_and_columns={
            "particlepool": "__all__",
        },
        indices=uid_common,
        sort=True,
    )
    particlepool = event_table["particlepool"]

    print(
        "cosmic: ",
        pk,
        "median num. particles making water-Cherenkov shower^{-1}:",
        np.median(particlepool["num_water_cherenkov"]),
    )

    passing_trigger_set = set(passing_trigger[pk]["uid"])

    RRR[pk] = {}
    path_template = opj(
        res.paths["plenoirf_dir"],
        "event_table",
        pk,
        "particles.map",
        "*.tar.gz",
    )
    for run_path in glob.glob(path_template):
        with corsika_primary.particles.ParticleEventTapeReader(
            run_path
        ) as run:
            for event in run:
                evth, parreader = event

                uid = irf.unique.make_uid(
                    run_id=int(run.runh[corsika_primary.I.RUNH.RUN_NUMBER]),
                    event_id=int(evth[corsika_primary.I.EVTH.EVENT_NUMBER]),
                )

                RRR[pk][uid] = {
                    "num_water_cer": 0,
                    "num_unknown": 0,
                    "num_gamma": 0,
                }
                for particle_block in parreader:
                    for particle_row in particle_block:
                        corsika_particle_id = (
                            corsika_primary.particles.decode_particle_id(
                                code=particle_row[
                                    corsika_primary.I.PARTICLE.CODE
                                ]
                            )
                        )

                        if zoo.has(corsika_particle_id):
                            momentum_GeV = np.array(
                                [
                                    particle_row[
                                        corsika_primary.I.PARTICLE.PX
                                    ],
                                    particle_row[
                                        corsika_primary.I.PARTICLE.PY
                                    ],
                                    particle_row[
                                        corsika_primary.I.PARTICLE.PZ
                                    ],
                                ]
                            )

                            pos_m = 1e-2 * np.array(
                                [
                                    particle_row[corsika_primary.I.PARTICLE.Y],
                                    particle_row[corsika_primary.I.PARTICLE.X],
                                ]
                            )

                            if np.linalg.norm(pos_m) <= radius_m:
                                if (
                                    corsika_particle_id
                                    == corsika_primary.particles.identification.PARTICLES[
                                        "gamma"
                                    ]
                                ):
                                    # gamma
                                    E_gamma_GeV = np.linalg.norm(momentum_GeV)
                                    if E_gamma_GeV > 100e6 * 1e-9:
                                        RRR[pk][uid]["num_water_cer"] += 1
                                        RRR[pk][uid]["num_gamma"] += 1
                                else:
                                    if zoo.cherenkov_emission(
                                        corsika_id=corsika_particle_id,
                                        momentum_GeV=momentum_GeV,
                                        medium_key="water",
                                    ):
                                        RRR[pk][uid]["num_water_cer"] += 1

                        else:
                            RRR[pk][uid]["num_unknown"] += 1

    OUT = {}
    for uid in RRR[pk]:
        if uid in passing_trigger_set:
            OUT[uid] = RRR[pk][uid]
    RRR[pk] = OUT

res.stop()
