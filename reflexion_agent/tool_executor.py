from collections import defaultdict
import json
from dotenv import load_dotenv
from typing import List
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from chains import parser
from reflexion_agent.schemas import AnswerQuestion, Reflection
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# How to inoke tool in lang graph

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tool_executor = ToolExecutor([tavily_tool]) # Executor enablees async batch execution

def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)

    ids=[]
    tool_invocations=[]

    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(ToolInvocation(
                tool="tavily_search_results_json",
                tool_input=query
            ))
            ids.append(parsed_call["id"])

    outputs = tool_executor.batch(tool_invocations) # Allows paralelll async calls with batch

    # Map each output to its corresponding ID and tool input
    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output

    # convert mapped outputs to ToolMessages object
    tool_messages = []
    for id_, mapped_output in outputs_map.items():
        tool_messages.append(ToolMessage(content=json.dumps(mapped_output), tool_call_id=id_))

    return tool_messages

if __name__ == '__main__':
    print('hello')

    human_message = HumanMessage(
        content="Write about AI-powered SOC / Autonomous soc problem domain, list startups that do that and raised capital"
    )

    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[ # Each element is a search quert in taverly. Can take this for my own usage
            "AI-powered SOC Startups funding",
            "AI soc problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="call_KpYHiCHFFEmLitHFVFhky1Ra"
    )

    raw_res = execute_tools(
        state = [
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name":AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id":"call_KpYHiCHFFEmLitHFVFhky1Ra"
                    }
                ]
            )
        ]
    )

    print(raw_res)