from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from tools import create_file

# ---- Agent State ----
class AgentState(TypedDict):
    messages: List

# ---- LLM ----
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
).bind_tools([create_file])

# ---- Agent Node ----
def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# ---- Tool Node ----
def tool_node(state: AgentState):
    last_message = state["messages"][-1]

    tool_calls = last_message.tool_calls
    results = []

    for call in tool_calls:
        if call["name"] == "create_file":
            result = create_file.invoke(call["args"])
            results.append(
                ToolMessage(
                    content=result,
                    tool_call_id=call["id"]
                )
            )

    return {"messages": state["messages"] + results}

# ---- Routing Logic ----
def should_use_tool(state: AgentState):
    last_message = state["messages"][-1]
    return "tool_calls" in last_message.additional_kwargs

# ---- Build Graph ----
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_use_tool,
    {
        True: "tool",
        False: END
    }
)

graph.add_edge("tool", END)

agent = graph.compile()