import numpy as np

from dcrh.core.boundaries import StatisticalBlock
from dcrh.core.reference import EmpiricalReference
from dcrh.core.sequential import CUSUMState, DualChannelController


def test_cusum_onset_after_minimum():
    state = CUSUMState()
    state.update(-2.0, block_index=1)
    assert state.score == 0.0
    state.update(1.5, block_index=2)
    state.update(1.5, block_index=3)
    assert state.score == 3.0
    assert state.onset_block == 2


def _ref():
    return EmpiricalReference(
        delta_mean=np.asarray([-1.0, 0.0, 1.0]),
        delta_quantile=np.asarray([-1.0, 0.0, 1.0]),
        grounding=np.asarray([-1.0, 0.0, 1.0]),
        token_mass=2,
        metadata={},
    )


def _block(index: int, mean: float, quantile: float) -> StatisticalBlock:
    return StatisticalBlock(
        block_index=index,
        start_token_index=(index - 1) * 2,
        end_token_index=index * 2,
        token_count=2,
        entropy_mean=mean,
        entropy_quantile=quantile,
        grounding=None,
        atomic_segments=1,
    )


def test_entropy_only_controller_ignores_missing_grounding():
    controller = DualChannelController(
        reference=_ref(),
        threshold=10.0,
        use_entropy=True,
        use_grounding=False,
    )
    first = controller.update(_block(1, 0.5, 0.5), allow_alarm=False)
    second = controller.update(_block(2, 0.8, 0.9), allow_alarm=False)
    assert first.p_grounding is None
    assert second.p_entropy is not None
    assert second.p_grounding is None
    assert controller.ready(minimum_blocks=2, epsilon=10.0)
