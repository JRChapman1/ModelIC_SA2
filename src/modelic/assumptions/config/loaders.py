import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Union

from modelic.core.curves import YieldCurve, IndexCurve
from modelic.core.enums import CurveKind, YieldSchema, IndexSchema
from modelic.core.compounding import fwd_to_df, zero_to_df

Schema = Literal["one_year_forwards", "maturity_zeros", "index_levels"]


@dataclass(frozen=True)
class CurveSpec:
    kind: Literal["yield", "index"]
    schema: Schema
    path: str
    name: str

def _validate_years(years: np.ndarray, start: int) -> None:
    if years.dtype.kind not in "iu":
        raise ValueError("Years column of input curves must be integers.")
    if years[0] != start:
        raise ValueError(f"Years column of input curve start must be equal to start. Expected {start}, got {years[0]}.")
    if not np.all(np.diff(years) == 1):
        raise ValueError("Years column of input curve must be consecutive with step 1.")

def load_curve(spec: CurveSpec) -> Union[YieldCurve, IndexCurve]:

    df = pd.read_csv(spec.path)
    years = df['year'].to_numpy(int)
    rates = df['rate'].to_numpy(float)

    if spec.kind == CurveKind.YIELD:
        _validate_years(years, 1)
        if spec.schema == YieldSchema.MATURITY_ZEROS:
            dvec = zero_to_df(years, rates)
        elif spec.schema == YieldSchema.ONE_YEAR_FORWARDS:
            dvec = fwd_to_df(years, rates)
        else:
            raise ValueError(f"Unknown schema: {spec.schema}")
        return YieldCurve(years, dvec, spec.name)
    elif spec.kind == CurveKind.INDEX:
        _validate_years(years, 0)
        if spec.schema == IndexSchema.INDEX_LEVELS:
            return IndexCurve(years, df, spec.name)
