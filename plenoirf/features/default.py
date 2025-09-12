def transformation_fx_median():
    return "quantile(x, 0.5)"


def transformation_fx_containment_percentile_90():
    return "quantile(x, 0.95) - quantile(x, 0.05)"
