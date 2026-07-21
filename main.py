import os
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, START, END

# Load environment variables (e.g. OPENAI_API_KEY) from .env file
load_dotenv()

# ==========================================
# 1. GLOBAL STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    """
    Represents the shared global state passed between the agents in the graph.
    """
    query: str                # The original user request / query
    plan: str                 # The list of data analysis steps created by the Planner
    code: str                 # The Pandas python code written by the Coder
    review_feedback: str      # Safety feedback/revisions from the Reviewer
    is_safe: bool             # A flag indicating if the code has passed safety checks
    iterations: int           # Keep track of refinement iterations (prevents infinite loops)
    execution_output: str     # The stdout output from running the generated code
    execution_error: str      # Any runtime error or traceback from execution
    chart_path: str           # The file path of any generated chart/plot


# ==========================================
# Pydantic Model for Reviewer Structured Output
# ==========================================
class SafetyReview(BaseModel):
    is_safe: bool = Field(description="Set to True if the code contains no security violations. False otherwise.")
    feedback: str = Field(description="Explanatory feedback on why the code is safe or what security violations were found.")


# ==========================================
# 2. AGENTS & NODE DEFINITIONS
# ==========================================

# Initialize the LLM (gpt-4o-mini as requested)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# A. PLANNER AGENT
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Data Analysis Planner. Given a user's data query, break it down into a clear step-by-step plan for pandas analysis. Focus on data cleaning, transformation, and aggregation steps needed."),
    ("user", "User Query: {query}")
])

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the user's data analysis request and creates a step-by-step plan.
    """
    print("\n--- [Planner Agent] Generating analysis steps... ---")
    planner_chain = planner_prompt | llm
    response = planner_chain.invoke({"query": state["query"]})
    return {"plan": response.content}


# B. CODER AGENT
coder_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Python programmer specialized in Pandas and Matplotlib. \
Given a plan and safety feedback (if any), write clean, executable Python/Pandas code that strictly follows ALL of the rules below.

RULES:
1. OUTPUT FORMAT: Wrap your entire code block inside markdown tags: ```python ... ```. No prose outside the code block.
2. CHART SAVING: If the plan involves any visualization, create the chart with Matplotlib and save it using `plt.savefig('chart.png')`. Never call `plt.show()` — it blocks execution.
3. STRICT DATA LIMITING — THIS IS MANDATORY: When generating ANY visualization or chart, you MUST limit the dataset to the Top 10 or Top 15 items BEFORE plotting. Use `.head(10)`, `.nlargest(10, 'column')`, or `.sort_values(...).head(10)` so that the chart axes remain clean and labels never overlap. Failure to limit the data is a critical error.
4. ENCODING: When reading or writing any file (pd.read_csv, to_csv, open), always pass `encoding='utf-8'` to prevent Unicode errors on Windows.
5. SAFE IMPORTS ONLY: Do not import os, subprocess, shutil, socket, requests, or any module that interacts with the system or network."""),
    ("user", "Plan:\n{plan}\n\nReviewer Feedback:\n{feedback}")
])

def coder_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates Pandas Python code based on the generated plan and reviewer feedback (if any).
    """
    print("\n--- [Coder Agent] Writing Pandas code... ---")
    coder_chain = coder_prompt | llm
    response = coder_chain.invoke({
        "plan": state["plan"],
        "feedback": state.get("review_feedback", "No feedback yet. This is the first draft.")
    })
    return {"code": response.content}


# C. REVIEWER AGENT
reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Python code safety auditor for a data-analysis application.

ALWAYS ALLOW (these are safe and expected — do NOT flag them):
- pandas operations: pd.read_csv(), pd.DataFrame(), df.to_csv(), df.groupby(), df.head(), df.describe(), df.plot(), etc.
- matplotlib/seaborn operations: plt.savefig(), plt.figure(), plt.show(), plt.bar(), plt.title(), etc.
- numpy/scipy operations: np.mean(), np.array(), etc.
- standard Python builtins: print(), len(), range(), zip(), enumerate(), str(), int(), float()
- local file reads of .csv or .xlsx files
- saving .png/.jpg chart files locally via plt.savefig()
- specifying encoding='utf-8' in file operations

FLAG AS UNSAFE (only these patterns are genuinely dangerous):
- os.system(), os.popen(), os.remove() on system paths, os.rmdir() on system paths
- subprocess.run(), subprocess.Popen(), subprocess.call()
- shutil.rmtree() or any deletion of directories
- socket, requests, urllib — any outbound network call
- eval() or exec() on externally-supplied or unknown input strings
- __import__() with dynamic module names
- open() in write mode ('w') on paths outside the current working directory

Your response must use the structured output format with:
- is_safe: true if none of the UNSAFE patterns are present, false otherwise
- feedback: one concise sentence. If safe, say 'Code is safe to execute.' If unsafe, name the exact violation found."""),
    ("user", "Code:\n{code}")
])

def reviewer_node(state: AgentState) -> Dict[str, Any]:
    """
    Audits the generated Pandas code for security/safety. Uses structured output for consistency.
    """
    print("\n--- [Reviewer Agent] Auditing code safety... ---")
    reviewer_chain = reviewer_prompt | llm.with_structured_output(SafetyReview)
    result = reviewer_chain.invoke({"code": state["code"]})
    
    print(f"Safety Check Passed: {result.is_safe}")
    print(f"Feedback: {result.feedback}")
    
    return {
        "is_safe": result.is_safe,
        "review_feedback": result.feedback,
        "iterations": state.get("iterations", 0) + 1
    }


# Helper to extract python code block
def extract_code(markdown_text: str) -> str:
    """
    Helper to extract raw Python code from markdown blocks (e.g. ```python ... ```)
    """
    if "```python" in markdown_text:
        return markdown_text.split("```python")[1].split("```")[0].strip()
    elif "```" in markdown_text:
        return markdown_text.split("```")[1].split("```")[0].strip()
    return markdown_text.strip()


# D. EXECUTOR NODE
def executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the generated code securely using PythonREPL.
    """
    print("\n--- [Executor Agent] Executing safe code... ---")
    raw_code = extract_code(state["code"])
    
    # 1. Record existing PNG files and their modification times before running
    png_before = {}
    for f in os.listdir():
        if f.endswith(".png"):
            try:
                png_before[f] = os.path.getmtime(f)
            except Exception:
                pass
                
    # Setup LangChain's PythonREPL
    repl = PythonREPL()
    
    try:
        output = repl.run(raw_code)
        
        # Safe print output to handle platform-specific encoding issues (e.g. Windows CP1252)
        try:
            print("Execution Output:\n", output)
        except UnicodeEncodeError:
            safe_output = output.encode('ascii', errors='replace').decode('ascii')
            print("Execution Output (Safe print fallback):\n", safe_output)
        
        # 2. Check for new or modified PNG files
        chart_val = ""
        for f in os.listdir():
            if f.endswith(".png"):
                try:
                    mtime = os.path.getmtime(f)
                    # If file is newly created or its modification time changed, it is our chart!
                    if f not in png_before or mtime > png_before[f]:
                        chart_val = f
                        break
                except Exception:
                    pass
                    
        # Check if the output contains a traceback error
        if "Traceback" in output or "Error:" in output:
            return {
                "execution_output": "",
                "execution_error": output,
                "chart_path": ""
            }
        return {
            "execution_output": output,
            "execution_error": "",
            "chart_path": chart_val
        }
    except Exception as e:
        error_msg = str(e)
        try:
            print("Execution Error:\n", error_msg)
        except UnicodeEncodeError:
            safe_error_msg = error_msg.encode('ascii', errors='replace').decode('ascii')
            print("Execution Error (Safe print fallback):\n", safe_error_msg)
            
        return {
            "execution_output": "",
            "execution_error": error_msg,
            "chart_path": ""
        }


# ==========================================
# 3. ROUTING LOGIC
# ==========================================
def should_continue(state: AgentState) -> str:
    """
    Determines whether to finish the workflow or route back to the Coder Agent for revisions.
    """
    if state["is_safe"]:
        print("--> Proceeding to execution/end.")
        return "safe"
    elif state.get("iterations", 0) >= 3:
        print("--> Reached max revision iterations (3). Stopping workflow with error.")
        return "max_retries"
    else:
        print("--> Code failed safety audit. Re-routing back to Coder Agent for revisions.")
        return "unsafe"


# ==========================================
# 4. LANGGRAPH WORKFLOW CONFIGURATION
# ==========================================

# Initialize the StateGraph with our custom State representation
workflow = StateGraph(AgentState)

# Add Nodes representing the agent actions
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("executor", executor_node)

# Set the entrypoint
workflow.add_edge(START, "planner")

# Set static progression: planner -> coder -> reviewer
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "reviewer")

# Set conditional edge out of the reviewer node
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "safe": "executor",
        "max_retries": END,
        "unsafe": "coder"
    }
)

# Set progression from executor to end
workflow.add_edge("executor", END)

# Compile the workflow into a runnable app
app = workflow.compile()


# ==========================================
# EXAMPLE RUN
# ==========================================
if __name__ == "__main__":
    # Ensure you have your OPENAI_API_KEY configured in your environment or a .env file.
    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] OPENAI_API_KEY is not set in environment or .env file. Running might fail.")

    sample_query = "Load the file sales_data.csv, filter rows where Sales > 1000, and calculate the mean revenue by region."
    
    # Initialize the input state
    initial_state = {
        "query": sample_query,
        "plan": "",
        "code": "",
        "review_feedback": "",
        "is_safe": False,
        "iterations": 0,
        "execution_output": "",
        "execution_error": "",
        "chart_path": ""
    }
    
    print("=== Multi-Agent Data Analysis Workflow Started ===")
    final_output = app.invoke(initial_state)
    print("\n================== WORKFLOW COMPLETE ==================")
    print("\n--- FINAL PLAN ---")
    print(final_output["plan"])
    print("\n--- FINAL CODE ---")
    print(final_output["code"])
    print(f"\nFinal Safety Status: {final_output['is_safe']}")
    print(f"Final Feedback: {final_output['review_feedback']}")
    print("\n--- EXECUTION OUTPUT ---")
    print(final_output.get("execution_output", "No execution output."))
    if final_output.get("chart_path"):
        print(f"\nGenerated Chart Path: {final_output['chart_path']}")
    if final_output.get("execution_error"):
        print(f"\n--- EXECUTION ERROR ---")
        print(final_output["execution_error"])
