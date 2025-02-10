from typing import Any, Dict
from xml.dom.minidom import Document
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    
    print("---Checking Relevence of Documnet to Question---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"quesiton": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == 'yes':
            print("---GRADE: DOCUMENT RELEVENT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVENT---")
            web_search = True
            continue
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search}