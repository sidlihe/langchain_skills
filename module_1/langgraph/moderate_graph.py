import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END

load_dotenv()

class AgentState(TypedDict):
    messages: list
    intent: str

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

def classify_intent(state: AgentState) -> AgentState:
    user_message = state["messages"][-1].content

    if "hello" in user_message.lower():
        intent = "greeting"
    elif "python" in user_message.lower():
        intent = "technical"
    else:
        intent = "general"

    print("Detected Intent:", intent)

    return {**state, "intent": intent}

def greeting_node(state: AgentState) -> AgentState:
    response = "Hey! How can I assist you today?"

    return {
        "messages": state["messages"] + [AIMessage(content=response)],
        "intent": state["intent"],
    }

def technical_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {
        "messages": state["messages"] + [response],
        "intent": state["intent"],
    }

def general_node(state: AgentState) -> AgentState:
    response = "That's interesting! Let me think about it."
    return {
        "messages": state["messages"] + [AIMessage(content=response)],
        "intent": state["intent"],
    }

def route_intent(state: AgentState):
    return state["intent"]

builder = StateGraph(AgentState)

builder.add_node("classifier", classify_intent)
builder.add_node("greeting", greeting_node)
builder.add_node("technical", technical_node)
builder.add_node("general", general_node)
builder.set_entry_point("classifier")

builder.add_conditional_edges(
    "classifier",
    route_intent,
    {
        "greeting": "greeting",
        "technical": "technical",
        "general": "general",
    }
)

builder.add_edge("greeting", END)
builder.add_edge("technical", END)
builder.add_edge("general", END)

graph = builder.compile()

user_input = input("Ask something: ")

result = graph.invoke(
    {
        "messages": [HumanMessage(content=user_input)],
        "intent": "",
    }
)

print("\nFinal Output:")
print(result["messages"][-1].content)

