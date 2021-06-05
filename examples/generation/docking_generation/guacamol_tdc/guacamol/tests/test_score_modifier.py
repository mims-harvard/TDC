from functools import partial

import numpy as np
import pytest

from guacamol.score_modifier import LinearModifier, SquaredModifier, AbsoluteScoreModifier, GaussianModifier, \
    MinGaussianModifier, MaxGaussianModifier, ThresholdedLinearModifier, ClippedScoreModifier, \
    SmoothClippedScoreModifier, ChainedModifier

scalar_value = 8.343
value_array = np.array([[-3.3, 0, 5.5],
                        [0.011, 2.0, -33]])


def test_linear_function_default():
    f = LinearModifier()

    assert f(scalar_value) == scalar_value
    assert np.array_equal(f(value_array), value_array)


def test_linear_function_with_slope():
    slope = 3.3
    f = LinearModifier(slope=slope)

    assert f(scalar_value) == slope * scalar_value
    assert np.array_equal(f(value_array), slope * value_array)


def test_squared_function():
    target_value = 5.555
    coefficient = 0.123
    f = SquaredModifier(target_value=target_value, coefficient=coefficient)

    expected_scalar = 1.0 - coefficient * (target_value - scalar_value) ** 2
    expected_array = 1.0 - coefficient * np.square(target_value - value_array)

    assert f(scalar_value) == expected_scalar
    assert np.array_equal(f(value_array), expected_array)


def test_absolute_function():
    target_value = 5.555
    f = AbsoluteScoreModifier(target_value=target_value)

    expected_scalar = 1.0 - abs(target_value - scalar_value)
    expected_array = 1.0 - np.abs(target_value - value_array)

    assert f(scalar_value) == expected_scalar
    assert np.array_equal(f(value_array), expected_array)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def test_gaussian_function():
    mu = -1.223
    sigma = 0.334

    f = GaussianModifier(mu=mu, sigma=sigma)

    assert f(mu) == 1.0
    assert f(scalar_value) == gaussian(scalar_value, mu, sigma)
    assert np.allclose(f(value_array), gaussian(value_array, mu, sigma))


def test_min_gaussian_function():
    mu = -1.223
    sigma = 0.334

    f = MinGaussianModifier(mu=mu, sigma=sigma)

    assert f(mu) == 1.0

    low_value = -np.inf
    large_value = np.inf

    assert f(low_value) == 1.0
    assert f(large_value) == 0.0

    full_gaussian = partial(gaussian, mu=mu, sig=sigma)
    min_gaussian_lambda = lambda x: 1.0 if x < mu else full_gaussian(x)
    min_gaussian = np.vectorize(min_gaussian_lambda)

    assert f(scalar_value) == min_gaussian(scalar_value)
    assert np.allclose(f(value_array), min_gaussian(value_array))


def test_max_gaussian_function():
    mu = -1.223
    sigma = 0.334

    f = MaxGaussianModifier(mu=mu, sigma=sigma)

    assert f(mu) == 1.0

    low_value = -np.inf
    large_value = np.inf

    assert f(low_value) == 0.0
    assert f(large_value) == 1.0

    full_gaussian = partial(gaussian, mu=mu, sig=sigma)
    max_gaussian_lambda = lambda x: 1.0 if x > mu else full_gaussian(x)
    max_gaussian = np.vectorize(max_gaussian_lambda)

    assert f(scalar_value) == max_gaussian(scalar_value)
    assert np.allclose(f(value_array), max_gaussian(value_array))


def test_tanimoto_threshold_function():
    threshold = 5.555
    f = ThresholdedLinearModifier(threshold=threshold)

    large_value = np.inf

    assert f(large_value) == 1.0
    assert f(threshold) == 1.0

    expected_array = np.minimum(value_array, threshold) / threshold
    assert np.array_equal(f(value_array), expected_array)


def test_clipped_function():
    min_x = 4.4
    max_x = 8.8
    min_score = -3.3
    max_score = 9.2

    modifier = ClippedScoreModifier(upper_x=max_x, lower_x=min_x, high_score=max_score, low_score=min_score)

    # values smaller than min_x should be assigned min_score
    for x in [-2, 0, 4, 4.4]:
        assert modifier(x) == min_score

    # values larger than max_x should be assigned min_score
    for x in [8.8, 9.0, 1000]:
        assert modifier(x) == max_score

    # values in between are interpolated
    slope = (max_score - min_score) / (max_x - min_x)
    for x in [4.4, 4.8, 5.353, 8.034, 8.8]:
        dx = x - min_x
        dy = dx * slope
        assert modifier(x) == pytest.approx(min_score + dy)


def test_clipped_function_inverted():
    # The clipped function also works for decreasing scores
    max_x = 4.4
    min_x = 8.8
    min_score = -3.3
    max_score = 9.2

    modifier = ClippedScoreModifier(upper_x=max_x, lower_x=min_x, high_score=max_score, low_score=min_score)

    # values smaller than max_x should be assigned the maximal score
    for x in [-2, 0, 4, 4.4]:
        assert modifier(x) == max_score

    # values larger than min_x should be assigned min_score
    for x in [8.8, 9.0, 1000]:
        assert modifier(x) == min_score

    # values in between are interpolated
    slope = (max_score - min_score) / (max_x - min_x)
    for x in [4.4, 4.8, 5.353, 8.034, 8.8]:
        dx = x - min_x
        dy = dx * slope
        assert modifier(x) == pytest.approx(min_score + dy)


def test_thresholded_is_special_case_of_clipped_for_positive_input():
    threshold = 4.584
    thresholded_modifier = ThresholdedLinearModifier(threshold=threshold)
    clipped_modifier = ClippedScoreModifier(upper_x=threshold)

    values = np.array([0, 2.3, 8.545, 3.23, 0.12, 55.555])

    assert np.allclose(thresholded_modifier(values), clipped_modifier(values))


def test_smooth_clipped():
    min_x = 4.4
    max_x = 8.8
    min_score = -3.3
    max_score = 9.2

    modifier = SmoothClippedScoreModifier(upper_x=max_x, lower_x=min_x, high_score=max_score, low_score=min_score)

    # assert that the slope in the middle is correct

    middle_x = (min_x + max_x) / 2
    delta = 1e-5
    vp = modifier(middle_x + delta)
    vm = modifier(middle_x - delta)

    slope = (vp - vm) / (2 * delta)
    expected_slope = (max_score - min_score) / (max_x - min_x)

    assert slope == pytest.approx(expected_slope)

    # assert behavior at +- infinity

    assert modifier(1e5) == pytest.approx(max_score)
    assert modifier(-1e5) == pytest.approx(min_score)


def test_smooth_clipped_inverted():
    # The smooth clipped function also works for decreasing scores
    max_x = 4.4
    min_x = 8.8
    min_score = -3.3
    max_score = 9.2

    modifier = SmoothClippedScoreModifier(upper_x=max_x, lower_x=min_x, high_score=max_score, low_score=min_score)

    # assert that the slope in the middle is correct

    middle_x = (min_x + max_x) / 2
    delta = 1e-5
    vp = modifier(middle_x + delta)
    vm = modifier(middle_x - delta)

    slope = (vp - vm) / (2 * delta)
    expected_slope = (max_score - min_score) / (max_x - min_x)

    assert slope == pytest.approx(expected_slope)

    # assert behavior at +- infinity

    assert modifier(1e5) == pytest.approx(min_score)
    assert modifier(-1e5) == pytest.approx(max_score)


def test_chained_modifier():
    linear = LinearModifier(slope=2)
    squared = SquaredModifier(10.0)

    chained_1 = ChainedModifier([linear, squared])
    chained_2 = ChainedModifier([squared, linear])

    expected_1 = 1.0 - np.square(10.0 - (2 * scalar_value))
    expected_2 = 2 * (1.0 - np.square(10.0 - scalar_value))

    assert chained_1(scalar_value) == expected_1
    assert chained_2(scalar_value) == expected_2
