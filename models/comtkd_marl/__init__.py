"""CoMTKD-MARL model components."""
from .actor import TeacherActor, CardinalityPolicy
from .cardinality import estimate_optimal_cardinality, select_topk_mask
from .controller import CoMTKDMARL, JointPolicyOutput
from .critic import CentralizedCoherenceCritic
from .observations import OBSERVATION_NAMES, build_teacher_observations
from .ppo import MAPPOTrainer, RolloutBuffer
from .synchronization import (
    KnowledgeSynchronizationOracle,
    pairwise_js_divergence,
    probability_coherence_index,
)

__all__ = [
    "TeacherActor",
    "CardinalityPolicy",
    "CoMTKDMARL",
    "JointPolicyOutput",
    "CentralizedCoherenceCritic",
    "KnowledgeSynchronizationOracle",
    "OBSERVATION_NAMES",
    "build_teacher_observations",
    "MAPPOTrainer",
    "RolloutBuffer",
    "estimate_optimal_cardinality",
    "select_topk_mask",
    "pairwise_js_divergence",
    "probability_coherence_index",
]
