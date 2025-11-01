from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# 1. State (shared memory shape)
class State(dict):
    pass

# 2. Model
llm = ChatOpenAI(model="gpt-4.1")

# 3. Node = action (step)
def agent_node(state: State):
    user_input = state["input"]
    response = llm.invoke(user_input)
    return {"output": response.content}

# 4. Build graph
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")

# 5. Compile
app = graph.compile()

# Run it
result = app.invoke({"input": "Explain AI like I'm 10"})
print(result["output"])