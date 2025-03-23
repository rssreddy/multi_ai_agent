from typing import Literal
from pydantic import BaseModel, Field

class Supervisor(BaseModel):
    """Model for supervisor decisions on routing workflow."""
    next: Literal["enhancer", "researcher", "coder"] = Field(
        description="Specifies the next worker in the pipeline: "
                    "'enhancer' for enhancing the user prompt if it is unclear or vague, "
                    "'researcher' for additional information gathering, "
                    "'coder' for solving technical or code-related problems."
    )
    reason: str = Field(
        description="The reason for the decision, providing context on why a particular worker was chosen."
    )

class Validator(BaseModel):
    """Model for validator decisions on workflow completion."""
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate."
    )
    reason: str = Field(
        description="The reason for the decision."
    )