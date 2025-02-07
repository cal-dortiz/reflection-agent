import operator
from typing import Annotated, TypedDict, Union
from langchain_core.agents import AgentAction, AgentFinish

#Dive deeper into this as a way to save state.
class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]