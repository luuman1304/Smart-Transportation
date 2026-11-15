import pytest

from lastmile_data.train_sarp_phase_b import finite_sample_upper_quantile, rank_correlation


def test_finite_sample_upper_quantile_uses_conservative_ceiling_rank():
    # ceil((4 + 1) * .8) = 4: the maximum calibration residual.
    assert finite_sample_upper_quantile([3, 1, 4, 2], alpha=.2) == 4
    assert finite_sample_upper_quantile([1] * 12, alpha=.1) == 1


def test_finite_sample_upper_quantile_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        finite_sample_upper_quantile([])
    with pytest.raises(ValueError):
        finite_sample_upper_quantile([1], alpha=1)


def test_rank_correlation_handles_order_ties_and_constants():
    assert rank_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1)
    assert rank_correlation([1, 2, 3], [30, 20, 10]) == pytest.approx(-1)
    assert rank_correlation([1, 1, 1], [1, 2, 3]) == 0
