from typing import Dict, List, Optional
from pydantic import BaseModel

class TaskQuery(BaseModel):
    task_id: str
    task_description: dict
    task_type: str

class ReviewQuery(BaseModel):
    task_id: str
    file_path: str
    code_content: str
    requirements: List[str]
    acceptance_criteria: List[str]

class Query(BaseModel):
    task_id: str
    task_type: str
    task_description: Dict

class CodeGenerationResponse(BaseModel):
    file_path: str
    code: str
    message: str
    status: str

class ReviewResponse(BaseModel):
    file_path: str
    issues: List[str]
    suggestions: List[str]
    status: str
    passed: bool 