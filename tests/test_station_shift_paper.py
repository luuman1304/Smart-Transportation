from lastmile_data.build_station_shift_paper import exact_sign_flip_test, exact_sign_test


def test_exact_sign_test_all_positive():
    result = exact_sign_test([1] * 5)
    assert result["positive_stations"] == 5
    assert result["two_sided_p"] == 2 / 32


def test_exact_sign_flip_test_all_equal_positive():
    result = exact_sign_flip_test([1, 1, 1])
    assert result["permutations"] == 8
    assert result["two_sided_p"] == 2 / 8
