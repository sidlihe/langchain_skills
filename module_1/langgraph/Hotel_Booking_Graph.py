from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import os

from dotenv import load_dotenv
load_dotenv()
# ----------------------------
# 1️⃣ Structured Output Schema
# ----------------------------

class BookingInfo(BaseModel):
    intent: str = Field(description="User intent like book_hotel or other")
    city: Optional[str] = Field(description="City name")
    date: Optional[str] = Field(description="Booking date")
    room_type: Optional[str] = Field(description="Room type requested")


# ----------------------------
# 2️⃣ Graph State
# ----------------------------

class BookingState(TypedDict):
    messages: List
    booking_info: Optional[BookingInfo]
    available: Optional[bool]
    booking_confirmed: Optional[bool]


# ----------------------------
# 3️⃣ Groq LLM Setup
# ----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

structured_llm = llm.with_structured_output(BookingInfo)


# ----------------------------
# 4️⃣ Extract Booking Info (LLM)
# ----------------------------

def extract_info(state: BookingState) -> BookingState:
    user_msg = state["messages"][-1].content
    result = structured_llm.invoke(user_msg)

    print("🔍 Extracted:", result)

    return {
        **state,
        "booking_info": result
    }


# ----------------------------
# 5️⃣ Validate Required Fields
# ----------------------------

def validate_info(state: BookingState):
    info = state["booking_info"]

    if info.intent != "book_hotel":
        return "not_booking"

    if not info.city or not info.date or not info.room_type:
        return "missing_info"

    return "complete"


# ----------------------------
# 6️⃣ Ask Missing Info
# ----------------------------

def ask_missing(state: BookingState) -> BookingState:
    return {
        **state,
        "messages": state["messages"] + [
            AIMessage(content="Please provide city, date, and room type.")
        ]
    }


# ----------------------------
# 7️⃣ Handle Non Booking Query
# ----------------------------

def not_booking_response(state: BookingState) -> BookingState:
    return {
        **state,
        "messages": state["messages"] + [
            AIMessage(content="I currently handle only hotel bookings.")
        ]
    }


# ----------------------------
# 8️⃣ Simulated Availability Check (Tool Layer)
# ----------------------------

def check_availability(state: BookingState) -> BookingState:
    info = state["booking_info"]

    available = (
        info.city.lower() == "mumbai"
        and info.room_type.lower() == "deluxe"
    )

    print("🏨 Available:", available)

    return {
        **state,
        "available": available
    }


# ----------------------------
# 9️⃣ Final Response
# ----------------------------

def final_response(state: BookingState) -> BookingState:
    info = state["booking_info"]

    if state["available"]:
        message = f"✅ Your {info.room_type} room in {info.city} is booked for {info.date}."
        confirmed = True
    else:
        message = "❌ Sorry, room not available."
        confirmed = False

    return {
        **state,
        "booking_confirmed": confirmed,
        "messages": state["messages"] + [AIMessage(content=message)]
    }


# ----------------------------
# 🔟 Build LangGraph
# ----------------------------

builder = StateGraph(BookingState)

builder.add_node("extract", extract_info)
builder.add_node("ask_missing", ask_missing)
builder.add_node("not_booking", not_booking_response)
builder.add_node("check", check_availability)
builder.add_node("final", final_response)

builder.set_entry_point("extract")

builder.add_conditional_edges(
    "extract",
    validate_info,
    {
        "missing_info": "ask_missing",
        "complete": "check",
        "not_booking": "not_booking",
    }
)

builder.add_edge("check", "final")
builder.add_edge("ask_missing", END)
builder.add_edge("not_booking", END)
builder.add_edge("final", END)

graph = builder.compile()


# ----------------------------
# 🚀 Run Example
# ----------------------------

result = graph.invoke({
    "messages": [
        HumanMessage(content="Book deluxe room in Mumbai for tomorrow")
    ],
    "booking_info": None,
    "available": None,
    "booking_confirmed": None,
})

print("\n🟢 Final Output:")
print(result["messages"][-1].content)