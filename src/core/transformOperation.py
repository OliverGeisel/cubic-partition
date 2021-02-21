from enum import Enum


class TransformationOperation(Enum):
    NOT_SPECIFIC = -1
    MANUAL = 0
    ADD = 1
    REMOVE = 2
    SPLIT = 3
    MOVE_X = 4
    MOVE_5 = 5
    ITERATE = 6
    REDUCE = 7
