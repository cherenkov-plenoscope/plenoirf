import os
from os import path as op
from os.path import join as opj
import shutil
import subprocess
import importlib
from importlib import resources

import plenopy
import json_line_logger


def make_write_and_plot_sum_trigger_geometry(
    trigger_geometry_path,
    sum_trigger_config,
    light_field_calibration_path,
    logger=None,
):
    """
    Estimate the trigger-geometry of the instrument's sum-trigger.
    This geometry is about which read-out channel will be added to what pixel
    of the trigger's image.

    Parameters
    ----------
    trigger_geometry_path : str
        Path to the output zipfile where the trigger-geometry is written to.
    sum_trigger_config : dict
        Contains the the geometry of the trigger's image. Also contains the
        objectdistances to focus on.
    light_field_calibration_path : str
        Path to the light-field calibration of the instrument.
    logger : e.g. logging.Logger()
        Optional logger. If None, log will go to stdout.
    """
    if logger is None:
        logger = json_line_logger.LoggerStdout()

    logger.info("Estimating trigger-geometry.")
    if not op.exists(trigger_geometry_path):
        logger.info("read light_field_geometry.")
        light_field_geometry = plenopy.LightFieldGeometry(
            path=light_field_calibration_path
        )

        logger.info("make trigger_image_geometry.")
        trigger_image_geometry = (
            plenopy.trigger.geometry.init_trigger_image_geometry(
                **sum_trigger_config["image"]
            )
        )

        logger.info("make trigger_geometry.")
        trigger_geometry = plenopy.trigger.geometry.init_trigger_geometry(
            light_field_geometry=light_field_geometry,
            trigger_image_geometry=trigger_image_geometry,
            object_distances=sum_trigger_config["object_distances_m"],
        )

        logger.info(
            "write trigger_geometry to {:s}.".format(trigger_geometry_path)
        )
        plenopy.trigger.geometry.write(
            trigger_geometry=trigger_geometry,
            path=trigger_geometry_path,
        )

        logger.info("plotting trigger_geometry...")
        try:
            plots_dir = trigger_geometry_path + ".plots"
            plenopy.trigger.geometry.plot(
                trigger_geometry_path=trigger_geometry_path,
                out_dir=plots_dir,
            )
            shutil.make_archive(plots_dir, format="zip", root_dir=plots_dir)
            shutil.rmtree(plots_dir)
            logger.info("plotting trigger_geometry done.")
        except:
            logger.info("plotting trigger_geometry failed.")

    else:
        logger.info("Already done. Nothing to do.")
