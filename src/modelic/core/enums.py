from enum import Enum
from dataclasses import dataclass


@dataclass
class CurveKind:
    YIELD = "yield"
    INDEX = "index"

@dataclass
class YieldSchema(str, Enum):
    ONE_YEAR_FORWARDS = "one_year_forwards"
    MATURITY_ZEROS = "maturity_zeros"

@dataclass
class IndexSchema(str, Enum):
    INDEX_LEVELS = "index_levels"
