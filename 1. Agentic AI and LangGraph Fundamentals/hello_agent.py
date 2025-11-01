from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

# Load API keys from .env file (OPENAI_API_KEY)
load_dotenv()

# === 1. Define State ===
# State is like shared memory - it holds data that passes between steps
class State(TypedDict):
    input: str   # User's question
    output: str  # AI's response

# === 2. Initialize LLM ===
# Create an OpenAI chat model instance
llm = ChatOpenAI(model="gpt-4.1")

# === 3. Create Node Function ===
# A node is a function (operation) that performs work in your graph
# It receives the current state and returns updates to that state
# (In more complex agentic workflows, there will actions like tool use, web search, etc. within nodes)
def agent_node(state: State):
    user_input = state["input"]              # Get user's question from state
    response = llm.invoke(user_input)        # Send to OpenAI and get response
    return {"output": response.content}      # Return the AI's answer

# === 4. Build the Graph ===
# StateGraph manages the workflow and state transitions
graph = StateGraph(State)
graph.add_node("agent", agent_node)  # Add our function as a node
graph.set_entry_point("agent")       # Start execution from this node

# === 5. Compile and Run ===
# Compile converts the graph into an executable application
app = graph.compile()

# Invoke the app with initial state
result = app.invoke({"input": "Explain AI like I'm 10 in 2 short sentences.", "output": ""})
print(result["output"])