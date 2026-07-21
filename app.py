import streamlit as st
import os
import pandas as pd
from main import app  # Import the compiled LangGraph StateGraph workflow

def generate_suggestions(df):
    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    
    suggestions = []
    # Suggestion 1: General summary
    suggestions.append("Provide a summary analysis and description of the dataset.")
    
    # Suggestion 2: Aggregation by category if exists
    if cat_cols and num_cols:
        suggestions.append(f"Show the average {num_cols[0]} grouped by {cat_cols[0]}.")
    elif num_cols:
        suggestions.append("Show the average values for all numeric columns.")
    else:
        suggestions.append(f"Provide a frequency count of {cols[0]}.")
        
    # Suggestion 3: Top records by numeric column
    if num_cols:
        suggestions.append(f"Find the top 5 records with the highest {num_cols[0]}.")
    else:
        suggestions.append("Inspect the first 10 rows of the dataset.")
        
    return suggestions

# Set premium page layout configuration
st.set_page_config(
    page_title="Multi-Agent Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via markdown
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .assistant-status {
        font-size: 0.9rem;
        color: #4B5563;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div style='text-align: left; padding: 1rem 0rem 2rem 0rem; border-bottom: 1px solid #E5E7EB;'>
    <h1 style='color: #1E3A8A; font-family: sans-serif; margin-bottom: 0.2rem;'>🤖 Multi-Agent Data Analyst</h1>
    <p style='color: #4B5563; font-size: 1.1rem;'>An autonomous LangGraph workflow that cleans, filters, aggregates, and runs analysis on your datasets securely.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR: DATASET UPLOADER & PREVIEW
# ==========================================
st.sidebar.markdown("""
<div style='padding-bottom: 1rem;'>
    <h2 style='color: #1E3A8A; font-family: sans-serif;'>📂 Data Source</h2>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file to analyze:",
    type=["csv"],
    help="Upload your dataset. The Multi-Agent system will plan and execute Python/Pandas operations on it."
)

active_df = None
active_filename = None

if uploaded_file is not None:
    try:
        active_df = pd.read_csv(uploaded_file)
        active_filename = uploaded_file.name
        
        # Save file locally with its original name in the workspace directory.
        saved_path = os.path.join(os.getcwd(), active_filename)
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.sidebar.success(f"Successfully loaded '{active_filename}'")
    except Exception as e:
        st.sidebar.error(f"Error loading CSV file: {e}")
elif os.path.exists("sales_data.csv"):
    if st.sidebar.checkbox("Use sample dataset (sales_data.csv)", value=True):
        try:
            active_df = pd.read_csv("sales_data.csv")
            active_filename = "sales_data.csv"
            st.sidebar.info("Using default 'sales_data.csv' sample dataset.")
        except Exception as e:
            st.sidebar.error(f"Error loading sample dataset: {e}")

if active_df is not None:
    # Render a data preview and metadata
    st.sidebar.markdown("### 📊 Dataset Preview")
    st.sidebar.dataframe(active_df.head(5), use_container_width=True)
    
    # Summary metrics
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Rows", f"{active_df.shape[0]:,}")
    col2.metric("Columns", f"{active_df.shape[1]}")
    
    st.sidebar.markdown("**Column Names & Types:**")
    col_types = pd.DataFrame(active_df.dtypes.astype(str), columns=["Type"])
    st.sidebar.dataframe(col_types, use_container_width=True)
    
    st.sidebar.markdown("### 📊 Visualization Option")
    selected_viz = st.sidebar.radio(
        "Choose preferred chart:",
        ["None (Table Only)", "📊 Bar Chart", "📈 Line Chart", "Scatter Plot", "🥧 Pie Chart"],
        index=0
    )
else:
    st.sidebar.info("Please upload a CSV dataset to get started.")
    selected_viz = "None (Table Only)"


# ==========================================
# CHAT INTERFACE: CHAT HISTORY STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Upload a CSV file in the sidebar and ask me to analyze it. I will generate a step-by-step plan, write the Pandas code, review it for safety, and run the computations to show you the output!"
        }
    ]

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If the message contains additional agent artifacts, render them in logical containers
        if "plan" in msg and msg["plan"]:
            with st.expander("📋 View Analysis Plan"):
                st.markdown(msg["plan"])
        if "code" in msg and msg["code"]:
            with st.expander("💻 View Generated Code"):
                st.code(msg["code"], language="python")
        if "execution_output" in msg and msg["execution_output"]:
            st.markdown("### ⚙️ Run Execution Output")
            st.code(msg["execution_output"])
        if "chart_path" in msg and msg["chart_path"] and os.path.exists(msg["chart_path"]):
            st.markdown("### 📊 Generated Visualization")
            st.image(msg["chart_path"], use_container_width=True)
        if "debug_state" in msg and msg["debug_state"]:
            with st.expander("🛠️ View AgentState Debug Log"):
                st.json(msg["debug_state"])
        if "execution_error" in msg and msg["execution_error"]:
            st.error(f"Execution Error: {msg['execution_error']}")


# ==========================================
# WORKFLOW EXECUTION ON USER INPUT
# ==========================================

run_query = None

# Show suggestions to user if dataset is pre-loaded and conversation history is empty
if active_df is not None and len(st.session_state.messages) <= 1:
    suggestions = generate_suggestions(active_df)
    st.markdown("💡 **Suggested Questions:**")
    cols = st.columns(len(suggestions))
    for idx, sug in enumerate(suggestions):
        if cols[idx].button(sug, key=f"sug_{idx}"):
            run_query = sug

if prompt := st.chat_input("What would you like me to analyze in this dataset?"):
    run_query = prompt

if run_query:
    # 1. Display user query in the chat
    st.chat_message("user").markdown(run_query)
    st.session_state.messages.append({"role": "user", "content": run_query})
    
    # 2. Check if a dataset is loaded
    if active_df is None:
        with st.chat_message("assistant"):
            warning_msg = "Please upload a CSV dataset in the sidebar first before executing queries."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    else:
        # 3. Initialize assistant response container
        with st.chat_message("assistant"):
            
            # Extract column list to ground planner and coder agents
            column_names = active_df.columns.tolist()
            
            # Append visualization suggestion to formatted query if selected
            viz_suffix = ""
            if selected_viz != "None (Table Only)":
                chart_name = selected_viz.split(" ", 1)[1] if " " in selected_viz else selected_viz
                viz_suffix = f"\nAdditionally, please create a {chart_name} of the results."
            
            # Format the query, explicitly listing available column names to guide the planner and coder
            formatted_query = (
                f"Analyze the uploaded file '{active_filename}'.\n"
                f"Available column names: {column_names}\n"
                f"Request: {run_query}"
                f"{viz_suffix}"
            )
            
            # Initialize state representation
            initial_state = {
                "query": formatted_query,
                "plan": "",
                "code": "",
                "review_feedback": "",
                "is_safe": False,
                "iterations": 0,
                "execution_output": "",
                "execution_error": "",
                "chart_path": ""
            }
            
            # Keep trace dictionary to collect graph results as we stream
            workflow_state = initial_state.copy()
            
            # Placeholder containers rendered once, filled progressively as nodes yield
            status_ph    = st.empty()   # Active-node status badge
            plan_ph      = st.empty()   # Plan (filled when planner yields)
            code_ph      = st.empty()   # Code (filled when coder yields)
            review_ph    = st.empty()   # Reviewer verdict
            output_ph    = st.empty()   # Execution stdout
            chart_ph     = st.empty()   # Generated chart image
            error_ph     = st.empty()   # Execution error (if any)
            debug_ph     = st.empty()   # Debug JSON (bottom)

            # --- NODE LABEL MAP ---
            NEXT_LABEL = {
                "planner":  "💻 Coder Agent running...",
                "coder":    "🛡️ Reviewer Agent running...",
                "reviewer": "⚙️ Executor Agent running...",
                "executor": "✅ Analysis complete!",
            }

            with status_ph.status("🤖 Planner Agent running...", expanded=True) as status_container:
                try:
                    for event in app.stream(initial_state):
                        for node_name, state_update in event.items():

                            # ── PLANNER ────────────────────────────────────────
                            if node_name == "planner":
                                if "plan" in state_update:
                                    workflow_state["plan"] = state_update["plan"]
                                # Render plan immediately inside a nice expander
                                with plan_ph.expander("📋 Analysis Plan", expanded=True):
                                    st.markdown(workflow_state["plan"])

                            # ── CODER ──────────────────────────────────────────
                            elif node_name == "coder":
                                if "code" in state_update:
                                    workflow_state["code"] = state_update["code"]
                                with code_ph.expander("💻 Generated Code", expanded=False):
                                    st.code(workflow_state["code"], language="python")

                            # ── REVIEWER ───────────────────────────────────────
                            elif node_name == "reviewer":
                                for key in ("review_feedback", "is_safe", "iterations"):
                                    if key in state_update:
                                        workflow_state[key] = state_update[key]
                                safety_passed = workflow_state.get("is_safe", False)
                                if safety_passed:
                                    review_ph.success("🛡️ **Reviewer:** Code approved — proceeding to execution.")
                                else:
                                    iters = workflow_state.get("iterations", 1)
                                    review_ph.warning(
                                        f"⚠️ **Reviewer (iteration {iters}):** Safety concern found — "
                                        f"requesting revision.\n\n> {workflow_state.get('review_feedback','')}"
                                    )

                            # ── EXECUTOR ────────────────────────────────────────
                            elif node_name == "executor":
                                for key in ("execution_output", "execution_error", "chart_path"):
                                    if key in state_update:
                                        workflow_state[key] = state_update[key]

                                if workflow_state.get("execution_output"):
                                    with output_ph.container():
                                        st.markdown("### ⚙️ Execution Output")
                                        st.code(workflow_state["execution_output"])

                                if workflow_state.get("chart_path") and os.path.exists(workflow_state["chart_path"]):
                                    with chart_ph.container():
                                        st.markdown("### 📊 Generated Visualization")
                                        st.image(workflow_state["chart_path"], use_container_width=True)

                                if workflow_state.get("execution_error"):
                                    error_ph.error(f"Execution Error:\n{workflow_state['execution_error']}")

                            # Advance status label to the next node immediately
                            next_label = NEXT_LABEL.get(node_name, "✅ Analysis complete!")
                            status_container.update(label=next_label, state="running")

                    status_container.update(label="✅ Analysis Complete!", state="complete", expanded=False)

                except Exception as err:
                    status_container.update(label="❌ Workflow failed!", state="error", expanded=False)
                    error_ph.error(f"Runtime error: {err}")
                    workflow_state["execution_error"] = str(err)

            # Debug state log (always at the bottom)
            with debug_ph.expander("🛠️ View AgentState Debug Log", expanded=False):
                st.json(workflow_state)

            # 5. Append assistant message & all artifacts to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Analysis complete! Check the plan, code, and execution output above.",
                "plan":             workflow_state.get("plan", ""),
                "code":             workflow_state.get("code", ""),
                "execution_output": workflow_state.get("execution_output", ""),
                "execution_error":  workflow_state.get("execution_error", ""),
                "chart_path":       workflow_state.get("chart_path", ""),
                "debug_state":      workflow_state
            })


