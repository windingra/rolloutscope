"""Built-in diagnostic rules."""

from .entropy_collapse import EntropyCollapseRule
from .group_difficulty import GroupDifficultyRule
from .invalid_rollout import InvalidRolloutRule
from .length_collapse import LengthCollapseRule
from .reward_hacking import RewardHackingRule
from .zero_advantage import ZeroAdvantageRule

#: Order matters only for display; findings are re-sorted by severity.
ALL_RULES = [
    GroupDifficultyRule(),
    ZeroAdvantageRule(),
    InvalidRolloutRule(),
    RewardHackingRule(),
    EntropyCollapseRule(),
    LengthCollapseRule(),
]

__all__ = [
    "ALL_RULES",
    "EntropyCollapseRule",
    "GroupDifficultyRule",
    "InvalidRolloutRule",
    "LengthCollapseRule",
    "RewardHackingRule",
    "ZeroAdvantageRule",
]
