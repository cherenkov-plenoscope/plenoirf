from . import combined_features
from .. import event_table

import warnings
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import x
from sympy import lambdify


def find_values_quantile_range(values, quantile_range):
    """
    Returns (start, stop) of values range while excluding outliers
    outside of the quantile-range.
    """
    start = quantile_range[0]
    stop = quantile_range[1]
    assert start >= 0.0
    assert start < 1.0
    assert stop > 0.0
    assert stop <= 1.0
    sorted_values = np.sort(values)
    num = len(values)
    start_idx = int(np.ceil(num * start))
    stop_idx = int(np.floor(num * stop))
    if start_idx >= num:
        start_idx = num - 1
    if stop_idx >= num:
        stop_idx = num - 1
    return sorted_values[start_idx], sorted_values[stop_idx]


def function_from_string(function_string="1*x"):
    expression = sympy.parsing.sympy_parser.parse_expr(function_string)
    function = sympy.lambdify(sympy.abc.x, expression)
    return function


def transform(feature_raw, transformation):
    # apply function
    func = function_from_string(function_string=transformation["function"])
    f_trans = func(feature_raw)

    # scale
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in divide"
        )
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in divide"
        )
        f_scaled = (f_trans - transformation["shift"]) / transformation[
            "scale"
        ]

    return f_scaled


def find_transformation(feature_raw, transformation_instruction):
    ti = transformation_instruction
    transformation = {}
    transformation["function"] = str(ti["function"])

    # apply function
    func = function_from_string(function_string=ti["function"])
    f_trans = func(feature_raw)

    # find quantile
    start, stop = find_values_quantile_range(
        values=f_trans,
        quantile_range=ti["quantile_range"],
    )
    transformation["quantile_range"] = ti["quantile_range"]
    transformation["range"] = [start, stop]
    mask_quanitle = np.logical_and(f_trans >= start, f_trans <= stop)

    # scale
    shift_func = function_from_string(function_string=ti["shift"])
    scale_func = function_from_string(function_string=ti["scale"])

    transformation["shift"] = shift_func(f_trans[mask_quanitle])
    transformation["scale"] = scale_func(f_trans[mask_quanitle])

    return transformation


def init_all_features_structure():
    original = event_table.structure.init_features_level_structure()
    combined = combined_features.init_combined_features_structure()

    out = {}
    for fk in original:
        out[fk] = dict(original[fk])
    for fk in combined:
        out[fk] = dict(combined[fk])

    transformed_feature_structure = {"transformed_features": {}}
    for fk in out:
        transformed_feature_structure["transformed_features"][fk] = {
            "dtype": "<f8"
        }
    return out
