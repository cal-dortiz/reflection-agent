import datetime
from dotenv import load_dotenv

# Why use this over the standard output parsers?
from isort import find_imports_in_stream
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputKeyToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(Tools=[AnswerQuestion])

actor_prompt_tempalte = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert researcher. Current time: {time}
            1. {first_introduction}
            2. Reflect and critique your answer. Be sever to maximize improvement.
            3. Recommend search queries to research information and improve your answer.,
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format."
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


first_responder_prompt_template = actor_prompt_tempalte.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

# TODO:
# Force llm to always use the answerquestion tool. I relly need to dive into this to review how this is working. Lecture 17
# Grounding of the LLM comes from the pydantic object
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """
    Revise your previous answer using the new information.
    You should use the previous critique to add important information to your answer.
    You must include numerical citations in your revised answer to ensure it cna be verified.
    Add a "References section to the bottom of your answer (which does not count towards the word limit). In form of:
        [1] https://example.com
        [2] https://example.com
    You should use the previous critique to remove superfluous information from your answer and make sure it is not more than 250 words long.
    """

# The tool binding enforces the schema to the revise anser
revisor = actor_prompt_tempalte.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == '__main__':
    human_message = HumanMessage(
        content="write about AI-powered SOC / autonomous soc problem domain, list startups that do that and raised capital.")
    
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tool=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    