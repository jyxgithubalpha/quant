"""
State update utility functions.
"""

from typing import List, Optional

import numpy as np

from ..configs import StatePolicyConfig
from ..results import SplitResult
from .policy import StatePolicyFactory, _result_feature_hash
from .rolling import RollingState


def update_state(
    prev_state: Optional[RollingState],
    split_result: SplitResult,
    config: StatePolicyConfig,
) -> RollingState:
    policy = StatePolicyFactory.create(config)
    return policy.update_state(prev_state or RollingState(), split_result)


def aggregate_bucket_results(
    results: List[SplitResult],
    config: StatePolicyConfig,
) -> np.ndarray:
    policy = StatePolicyFactory.create(config)
    return policy.aggregate_importance(results)


def update_state_from_bucket_results(
    prev_state: Optional[RollingState],
    results: List[SplitResult],
    config: StatePolicyConfig,
) -> RollingState:
    policy = StatePolicyFactory.create(config)
    agg_importance = policy.aggregate_importance(results)

    feature_names_hash = ""
    for result in results:
        candidate_hash = _result_feature_hash(result)
        if candidate_hash:
            feature_names_hash = candidate_hash
            break

    return policy.update_state_from_bucket(
        prev_state or RollingState(),
        agg_importance,
        feature_names_hash,
    )
