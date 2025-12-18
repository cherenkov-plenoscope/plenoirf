from plenoirf.utils import astronomic_magnitude_to_brightness
from plenoirf.utils import astronomic_brightness_to_magnitude
from numpy.testing import assert_almost_equal


def test_unity():
    assert_almost_equal(astronomic_magnitude_to_brightness(1.0), 1.0)
    assert_almost_equal(astronomic_brightness_to_magnitude(1.0), 1.0)


def test_wikipedia_cartoon():
    assert_almost_equal(
        astronomic_magnitude_to_brightness(1.0), 1.0, decimal=3
    )
    assert_almost_equal(
        astronomic_magnitude_to_brightness(1.5), 0.631, decimal=3
    )
    assert_almost_equal(
        astronomic_magnitude_to_brightness(2.0), 0.398, decimal=3
    )
    assert_almost_equal(
        astronomic_magnitude_to_brightness(2.5), 0.251, decimal=3
    )
    assert_almost_equal(
        astronomic_magnitude_to_brightness(3.0), 0.158, decimal=3
    )
    assert_almost_equal(
        astronomic_magnitude_to_brightness(3.5), 0.100, decimal=3
    )


def test_forth_and_back():
    magnitudes = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    brightnesses = [1.0, 0.631, 0.398, 0.251, 0.158, 0.10]

    for i in range(len(magnitudes)):
        magnitude = magnitudes[i]
        brightness = astronomic_magnitude_to_brightness(magnitude)
        assert_almost_equal(brightness, brightnesses[i], decimal=3)
        magnitude_back = astronomic_brightness_to_magnitude(brightness)
        assert_almost_equal(magnitude_back, magnitude, decimal=3)
