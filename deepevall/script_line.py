from pydantic import BaseModel, validator
from typing import Optional

class ScriptLine(BaseModel):
    text: Optional[str] = None
    speaker: Optional[str] = None

    @validator('text', 'speaker', pre=True, always=True)
    def strings_must_be_stripped(cls, v):
        return v.strip() if isinstance(v, str) else v