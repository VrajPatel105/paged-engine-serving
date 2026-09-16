from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Literal,  Annotated

class UserInput(BaseModel):

    prompt: Annotated[str, Field(..., min_length=1, max_length=200, description="user prompt / input")]
    max_tokens: Annotated[int, Field(25, ge=1, le=200, description="Maximum number of tokens ")]