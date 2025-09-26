from . import combined_features
from . import scaling
from .. import event_table

import warnings
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import x
from sympy import lambdify


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

    # scale
    shift_func = function_from_string(function_string=ti["shift"])
    scale_func = function_from_string(function_string=ti["scale"])

    transformation["shift"] = shift_func(f_trans)
    transformation["scale"] = scale_func(f_trans)

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
