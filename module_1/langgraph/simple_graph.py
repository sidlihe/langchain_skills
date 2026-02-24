import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END

load_dotenv()

# Define the structure of the graph's state

class AgentState(TypedDict):
    messages: list

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)


#Define Node
def chatbot_node(state:AgentState) -> AgentState:
    print("\n==========Node Executing==========")
    print("Current State:", state)

    response = llm.invoke(state["messages"])

    print("LLM Response:", response)
    return {
        "messages": state["messages"] + [response]
    }

# build graph
builder = StateGraph(AgentState)

builder.add_node("chatbot", chatbot_node)

builder.set_entry_point("chatbot")

builder.add_edge("chatbot", END)

graph = builder.compile()


# -----------------------------

user_input = input("Ask something: ")

result = graph.invoke(
    {
        "messages": [HumanMessage(content=user_input)]
    }
)

print("\n--- Final Output ---")
print(result["messages"][-1].content)
