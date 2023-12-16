#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import glob
import json_utils
import corsika_primary


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]

passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)

zoo = corsika_primary.particles.identification.Zoo(
    media_refractive_indices={"water": 1.33}
)


radius_m = 1e4

RRR = {}
for sk in ["chile"]:  # SITES:
    RRR[sk] = {}
    for pk in ["proton"]:  # PARTICLES:
        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
        )

        particlepool = spt.cut_level_on_indices(
            level=event_table["particlepool"],
            indices=passing_trigger[sk][pk]["idx"],
        )

        print(
            "site: ",
            sk,
            "cosmic: ",
            pk,
            "median num. particles making water-Cherenkov shower^{-1}:",
            np.median(particlepool["num_water_cherenkov"]),
        )

        passing_trigger_set = set(passing_trigger[sk][pk]["idx"])

        RRR[sk][pk] = {}
        path_template = os.path.join(
            pa["run_dir"], "event_table", sk, pk, "particles.map", "*.tar.gz"
        )
        for run_path in glob.glob(path_template):
            with corsika_primary.particles.ParticleEventTapeReader(
                run_path
            ) as run:
                for event in run:
                    evth, parreader = event

                    uid = irf.unique.make_uid(
                        run_id=int(
                            run.runh[corsika_primary.I.RUNH.RUN_NUMBER]
                        ),
                        event_id=int(
                            evth[corsika_primary.I.EVTH.EVENT_NUMBER]
                        ),
                    )

                    RRR[sk][pk][uid] = {
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
                                        particle_row[
                                            corsika_primary.I.PARTICLE.Y
                                        ],
                                        particle_row[
                                            corsika_primary.I.PARTICLE.X
                                        ],
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
                                        E_gamma_GeV = np.linalg.norm(
                                            momentum_GeV
                                        )
                                        if E_gamma_GeV > 100e6 * 1e-9:
                                            RRR[sk][pk][uid][
                                                "num_water_cer"
                                            ] += 1
                                            RRR[sk][pk][uid]["num_gamma"] += 1
                                    else:
                                        if zoo.cherenkov_emission(
                                            corsika_id=corsika_particle_id,
                                            momentum_GeV=momentum_GeV,
                                            medium_key="water",
                                        ):
                                            RRR[sk][pk][uid][
                                                "num_water_cer"
                                            ] += 1

                            else:
                                RRR[sk][pk][uid]["num_unknown"] += 1

        OUT = {}
        for uid in RRR[sk][pk]:
            if uid in passing_trigger_set:
                OUT[uid] = RRR[sk][pk][uid]
        RRR[sk][pk] = OUT
