from dotenv import load_dotenv
from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph
from chains import revisor, first_responder
from tool_executor import execute_tools

load_dotenv()


MAX_ITERATION = 2
builder = MessageGraph()
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)
builder.add_edge("draft","execute_tools")
builder.add_edge("execute_node","revise")

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state) # works because we are returnign a single tool message 
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATION:
        return END
    return "execute_tools"

builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")

graph = builder.compile()

print(graph.get_graph().draw_ascii())
print(graph.get_graph().draw_mermaid_png(output_file_path="graph.png"))

if __name__ == '__main__':
    print('hello')

    res = graph.invoke(
        "Write about AI-Powered SOC / autonomous soc provlem domain, list startups that have been funded."
    )

    print(res[-1].tool_calls[0]["args"]["answers"])


