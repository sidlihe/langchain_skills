import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent   # NEW LOCATION

load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a and return the result."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divide a by b and return the result. Returns 0 if b is 0."""
    if b == 0:
        return 0.0
    return a / b

tools = [add, subtract, multiply, divide]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # ✅ Updated model
    api_key=os.getenv("GROQ_API_KEY"),
)

agent = create_agent(
    model=llm,
    tools=tools,
)

user_input = input("Enter a math question (+,-,*,%) : (e.g., 'What is 50 / 10?'): ")
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }
)

print(response["messages"][-1].content)
# for msg in response["messages"]:
#     print("\n====================")
#     print("TYPE:", type(msg).__name__)
#     print(msg)