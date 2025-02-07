from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Crititque of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question")
    reflection: Reflection = Field(description="Your reflection on the initial answer.") # Dive into this ground llm based on respons. I think i do this bu with a more manual iplementation
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to adress the critique of your current answer."
    )

class ReviseAnswer(AnswerQuestion):
    references: List[str] = Field(description="Citations motivating your updated answer.")