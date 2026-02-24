Core Concepts of LangGraph

#---------------------------------
1. State

Shared memory across nodes.

Example:

class State(TypedDict):
    messages: list

State moves through the graph.

#----------------------------------
2. Node

A function that modifies state.

Example:

def call_model(state):
    ...
    return {"messages": [...]}

#----------------------------------
3. Edges

Define how flow moves:

*Linear
*Conditional
*Loop

#-----------------------------------
4. Router (Conditional Edge)

Decides next node dynamically.

Example:
If tool call → go to tool node

Else → end

#------------------------------------
5. END

Graph termination.