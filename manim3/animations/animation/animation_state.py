from enum import Enum


class AnimationState(Enum):
    UNBOUND = 0
    BEFORE_ANIMATION = 1
    ON_ANIMATION = 2
    AFTER_ANIMATION = 3
