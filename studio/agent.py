from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState

# Notice we don't need to define the State class anymore!
# We just use the built-in 'MessagesState'

def chatbot(state: MessagesState):
    llm = ChatOpenAI(model="gpt-4o") # Make sure this matches your OpenAI model
    # Invoke the LLM with the messages
    response = llm.invoke(state["messages"])
    # Return the new message
    return {"messages": [response]}

# Build Graph using MessagesState
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()