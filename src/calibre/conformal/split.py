"""Split: the settings of split calibration over a score window."""

from conformalforecast.nonconformity import AbsoluteResidual, SignedResidual
from pydantic import BaseModel, ConfigDict, Field


class Split(BaseModel):
    """The score and the number of latest known origins that a threshold reads."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    score: AbsoluteResidual | SignedResidual
    window: int = Field(ge=1)
