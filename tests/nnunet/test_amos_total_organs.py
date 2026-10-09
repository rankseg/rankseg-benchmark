"""Static safeguards for the optional official-checkpoint AMOS22 pilot."""

from pathlib import Path

import pytest

from rankseg_benchmark.monai.small_organs import ORGANS
from rankseg_benchmark.nnunet.amos_total_organs import MODEL_CHANNELS, select_cases


def test_total_segmentator_targets_are_the_same_six_amos_organs():
    assert MODEL_CHANNELS == {"left_adrenal": 9, "right_adrenal": 8,
                              "gallbladder": 4, "pancreas": 7,
                              "spleen": 1, "liver": 5}
    assert set(MODEL_CHANNELS) == {organ.name for organ in ORGANS}
    assert {organ.amos_label for organ in ORGANS} == {1, 4, 6, 10, 11, 12}


def test_pilot_selection_is_fixed_before_outcomes():
    cases = [(f"case_{i:03d}", Path(f"image_{i}"), Path(f"label_{i}"))
             for i in range(100)]
    selected = select_cases(cases, 10)
    assert [case_id for case_id, _, _ in selected] == [f"case_{i:03d}" for i in range(0, 100, 10)]
    with pytest.raises(ValueError, match="positive"):
        select_cases(cases, 0)
    with pytest.raises(ValueError, match="exactly ten"):
        select_cases(cases[:99], 10)
