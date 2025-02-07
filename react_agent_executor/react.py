from venv import create
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# Reasoning engine of our agent. Loop will implemented in langgraph
react_prompt: PromptTemplate = hub.pull("hqchase17/react")

@tool
def triple(num: float) -> float:
    """
    :param num: a number to tripple
    :return: the number tripled -> multiplied by 3
    """

    return 3 * float(num)

tools = [TavilySearchResults(max_results=1), triple]

llm = ChatOpenAI(model="gpt-4o-mini")

react_agent_runnable = create(llm, tools, react_prompt)