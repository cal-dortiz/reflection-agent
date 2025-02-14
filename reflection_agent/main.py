from email import message
from mailbox import Message
import os
from typing import List, Sequence
from annotated_types import Ge
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from reflection_agent.chains import generation_chain, reflection_chain
from dotenv import load_dotenv

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: Sequence[BaseMessage]):
    return generation_chain.invoke({"messages": state})

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflection_chain.invoke({"messages":messages})
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT,GENERATE)

graph = builder.compile()

print(graph.get_graph().draw_mermaid()) # Prints syntax for mermaid.live tool
print(graph.get_graph().draw_ascii()) # Get Ascii Drawing

if __name__ =='__main__':
    print('Hello LangGraph')
