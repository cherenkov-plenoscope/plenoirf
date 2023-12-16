from plenoirf.utils import latex_scientific


def test_format():
    txt = latex_scientific(0.0, format_template="{:.1e}")
    assert txt == r"0.0\times{}10^{0}"

    txt = latex_scientific(4.123, format_template="{:.1e}")
    assert txt == r"4.1\times{}10^{0}"

    txt = latex_scientific(4.123e12, format_template="{:.1e}")
    assert txt == r"4.1\times{}10^{12}"

    txt = latex_scientific(-4.123e12, format_template="{:.1e}")
    assert txt == r"-4.1\times{}10^{12}"

    txt = latex_scientific(-4.123e-12, format_template="{:.1e}")
    assert txt == r"-4.1\times{}10^{-12}"

    txt = latex_scientific(-4.123e-12, format_template="{:.6e}")
    assert txt == r"-4.123000\times{}10^{-12}"

    txt = latex_scientific(float("nan"), format_template="{:.6e}")
    assert txt == "nan"
