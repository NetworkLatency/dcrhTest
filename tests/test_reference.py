import numpy as np

from dcrh.core.reference import EmpiricalReference


def make_ref():
    return EmpiricalReference(
        delta_mean=np.array([-1.0, 0.0, 1.0]),
        delta_quantile=np.array([-1.0, 0.0, 1.0]),
        grounding=np.array([-2.0, 0.0, 2.0]),
        token_mass=10,
        metadata={},
    )


def test_tail_directions():
    ref = make_ref()
    _, _, p_high_entropy = ref.entropy_pvalue(10.0, 10.0)
    _, _, p_low_entropy = ref.entropy_pvalue(-10.0, -10.0)
    assert p_high_entropy < p_low_entropy
    assert ref.grounding_pvalue(-10.0) < ref.grounding_pvalue(10.0)


def test_entropy_only_reference_allows_empty_grounding():
    ref = EmpiricalReference(
        delta_mean=np.array([-1.0, 0.0, 1.0]),
        delta_quantile=np.array([-1.0, 0.0, 1.0]),
        grounding=np.array([]),
        token_mass=10,
        metadata={"channels": ["H"]},
    )
    _, _, p = ref.entropy_pvalue(0.0, 0.0)
    assert p > 0.0
