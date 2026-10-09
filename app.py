import os
import re
import json
import sqlite3
import traceback
from contextlib import closing
from datetime import date
from typing import TypedDict, List, Dict, Optional

import pandas as pd
import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kartify Support",
    page_icon="🛒",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kartify.db")

MAX_ATTEMPTS = 3          # total order_agent runs per turn (1 initial + 2 retries)
PASS_THRESHOLD = 0.75     # groundedness and precision must both reach this
RECURSION_LIMIT = 25      # worst case path uses ~13 steps
HISTORY_WINDOW = 6        # past exchanges sent to the LLM

FALLBACK_RESPONSE = (
    "Sorry, I couldn't process that request right now. "
    "Please try rephrasing your question."
)
ERROR_RESPONSE = (
    "Sorry, something went wrong on our side. Please try again in a moment."
)

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
def load_api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


OPENAI_API_KEY = load_api_key()
if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. Add it as an environment variable "
        "in your Render service settings and redeploy."
    )
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not os.path.exists(DB_PATH):
    st.error(f"Database file not found at {DB_PATH}. Make sure kartify.db is deployed next to app.py.")
    st.stop()

# ---------------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------------
@st.cache_resource
def load_llms():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60, max_retries=2)
    evaluate_llm = ChatOpenAI(model="gpt-4o", temperature=0, timeout=60, max_retries=2)
    return llm, evaluate_llm


llm, evaluate_llm = load_llms()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class OrderState(TypedDict, total=False):
    cust_id: str
    order_id: str
    order_context: str
    query: str
    final_response: str
    history: List[Dict[str, str]]
    intent: str
    evaluation: Dict[str, float]
    eval_feedback: str
    eval_decision: str
    retry_count: int
    best_response: str
    best_score: float
    guard_result: str
    conv_guard_result: str
    memory_saved: bool

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
class ConversationMemory:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add(self, msg: dict):
        self.history.append(msg)

    def get(self) -> List[Dict[str, str]]:
        return self.history

    def clear(self):
        self.history = []

# ---------------------------------------------------------------------------
# SQL tool
# ---------------------------------------------------------------------------
@tool
def fetch_order_details(order_id: str) -> str:
    """
    Fetch all order details for a given order_id from the Kartify database.
    Use this tool whenever the customer's query requires order-specific information.
    Returns a formatted string of order details, or an error message if not found.
    """
    oid = (order_id or "").strip()
    if not oid or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", oid):
        return f"Invalid order ID: '{order_id}'."
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM orders WHERE order_id = ?",
                conn,
                params=(oid,),
            )
        if df.empty:
            return f"No order found with ID {oid}."
        return df.to_string(index=False)
    except Exception as e:
        return f"Database error while fetching order {oid}: {e}"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Kartify Customer Service Agent. You help customers with questions about their orders.

You have access to the following tool:
  fetch_order_details(order_id): retrieves all order information from the database.

Follow the ReAct pattern:
  Thought: <your reasoning about what to do next>
  Action: fetch_order_details with the order_id from the customer's query
  Observation: <tool result>
  Thought: <reason about the observation and form your answer>
  Final Answer: <short, polite, conversational reply, no greetings, no sign-off>

Policy rules (apply before writing Final Answer):
  - If actual_delivery is null the order has not arrived yet. Do not mention return/replacement eligibility.
  - Only mention return or replacement terms when the customer explicitly asks.
  - Never invent data. Only use what the tool returned.
  - Keep the Final Answer concise and empathetic.
  - Never reveal internal data fields or technical reasons in your reply (e.g. do not mention that actual_delivery is null or any other raw database values).
  - If a customer asks why their order hasn't arrived yet, only state that it is still on the way and share the expected delivery date. Never explain the technical reason behind the delay status.
  - Never promise or suggest an early delivery. Always communicate the expected delivery date as-is without implying it could arrive sooner.
  - If the order has not arrived by the expected delivery date, empathetically acknowledge the delay and advise the customer to wait a little longer. Do not speculate on reasons.
  - If the tool returns an error or no order, politely say you could not find the order details right now.

Answer Guidelines:
  - Only answer what is asked in the Query
  - Check the Previous conversation (if any) before generating the reply
  - Vague queries like "where is my order" or "status?" refer to the given Order ID
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def history_to_text(history: List[Dict[str, str]]) -> str:
    recent = (history or [])[-HISTORY_WINDOW:]
    if not recent:
        return "(none)"
    return "\n".join(f"User: {h['user']}\nAssistant: {h['assistant']}" for h in recent)


def clean_final_answer(text: str) -> str:
    text = (text or "").strip()
    marker = "final answer:"
    idx = text.lower().rfind(marker)
    if idx >= 0:
        return text[idx + len(marker):].strip()
    lines = [
        ln for ln in text.splitlines()
        if not re.match(r"^\s*(thought|action|observation)\s*:", ln, re.IGNORECASE)
    ]
    return "\n".join(lines).strip()


def extract_json_from_llm(text: str) -> Optional[dict]:
    text = (text or "").strip()
    candidates = []
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    candidates.append(text)
    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def to_score(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Order agent
# ---------------------------------------------------------------------------
def order_agent(query: str, order_id: str, history: list, feedback: str = "") -> tuple:
    today = date.today().strftime("%d %B %Y")

    # First call: tool is forced so order data is always fetched.
    llm_forced = llm.bind_tools([fetch_order_details], tool_choice="fetch_order_details")
    # Second call: tools disabled so the model must write the answer.
    llm_answer = llm.bind_tools([fetch_order_details], tool_choice="none")

    user_content = (
        f"Previous conversation:\n{history_to_text(history)}\n\n"
        f"Customer query: {query}\n"
        f"Order ID: {order_id}\n"
        f"Today's date: {today}"
    )
    if feedback:
        user_content += (
            f"\n\nYour previous answer to this query was rejected by a reviewer. "
            f"Reviewer feedback: {feedback}\n"
            f"Answer strictly from the tool data and address the query directly."
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    ai_msg = llm_forced.invoke(messages)
    messages.append(ai_msg)

    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    order_context = ""
    if tool_calls:
        for tc in tool_calls:
            # Always use the session's order_id so a model typo can't break the lookup.
            order_context = fetch_order_details.invoke({"order_id": order_id})
            messages.append(ToolMessage(content=order_context, tool_call_id=tc["id"]))
    else:
        order_context = fetch_order_details.invoke({"order_id": order_id})
        messages.append(HumanMessage(content=f"Observation (order details):\n{order_context}"))

    messages.append(HumanMessage(
        content="Using the observation above, write only the Final Answer for the customer."
    ))

    final_msg = llm_answer.invoke(messages)
    final_response = clean_final_answer(final_msg.content)
    if not final_response:
        final_response = FALLBACK_RESPONSE

    return order_context, final_response

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def user_input_node(state: OrderState):
    # Older LangGraph versions reject empty updates, so always write a key.
    return {"query": (state.get("query") or "").strip()}


def intent_node(state: OrderState):
    prompt = f"""You are an intent classifier for customer service queries. Classify the user's latest query into one of these categories.
Use the previous conversation for context: short follow-ups like "why?", "when?", "and the refund?" about the order are category 2.
Return ONLY the numeric ID (0, 1, 2, or 3). No explanation.

0 - Escalation: user is very angry/frustrated, wants a human now.
1 - Exit: user is ending the conversation ("Thanks", "Bye", "Resolved").
2 - Process: an order related query, including vague ones like "where is the order" or "status?".
3 - Random/Unrelated/Vulnerable: out-of-scope or potentially unsafe query.

Previous conversation:
{history_to_text(state.get('history', []))}

Latest query: {state['query']}"""
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        m = re.search(r"[0-3]", result)
        intent = m.group(0) if m else "2"
    except Exception:
        traceback.print_exc()
        intent = "2"
    return {"intent": intent}


def router_node(state: OrderState):
    return "order_agent" if state.get("intent") == "2" else "exit_node"


def order_agent_node(state: OrderState):
    order_context, final_response = order_agent(
        query=state["query"],
        order_id=state["order_id"],
        history=state.get("history", []),
        feedback=state.get("eval_feedback", ""),
    )
    return {
        "order_context": order_context,
        "final_response": final_response,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def evaluation_node(state: OrderState):
    today = date.today().strftime("%d %B %Y")
    prompt = f"""Evaluate the assistant's response to a customer query using the provided order context.

Today's date: {today}
Context: {state.get('order_context', '')}
Query: {state['query']}
Response: {state.get('final_response', '')}

Instructions:
1. groundedness (0.0 to 1.0): how well the response is factually supported by the context.
   Close to 1 if all facts come from the context. Close to 0 if anything is fabricated.
   Saying the order could not be found is grounded when the context shows an error or no order.
2. precision (0.0 to 1.0): how directly the response addresses the query.
   Close to 1 if concise and focused. Close to 0 if it has irrelevant details or misses the point.
3. feedback: one short sentence on what to fix (empty string if nothing).

Return ONLY a JSON object:
{{"groundedness": float, "precision": float, "feedback": string}}
"""
    attempts = state.get("retry_count", 0)
    current = state.get("final_response", "")

    try:
        raw = evaluate_llm.bind(response_format={"type": "json_object"}).invoke(
            [HumanMessage(content=prompt)]
        ).content
        data = extract_json_from_llm(raw)
    except Exception:
        traceback.print_exc()
        data = None

    # Evaluator unavailable: accept the answer instead of looping.
    if not data:
        return {"evaluation": {}, "eval_feedback": "", "eval_decision": "accept"}

    g = to_score(data.get("groundedness"))
    p = to_score(data.get("precision"))
    score = min(g, p)
    feedback = str(data.get("feedback", "") or "")

    update: Dict = {
        "evaluation": {"groundedness": g, "precision": p},
        "eval_feedback": feedback,
    }

    best_score = state.get("best_score", -1.0)
    best_response = state.get("best_response", "")
    if score > best_score:
        best_score, best_response = score, current
        update["best_score"] = best_score
        update["best_response"] = best_response

    if g >= PASS_THRESHOLD and p >= PASS_THRESHOLD:
        update["eval_decision"] = "accept"
    elif attempts >= MAX_ATTEMPTS:
        # Out of retries: ship the best answer seen this turn.
        update["eval_decision"] = "accept"
        update["final_response"] = best_response or current
    else:
        update["eval_decision"] = "retry"
    return update


def retry_router(state: OrderState):
    return "order_agent" if state.get("eval_decision") == "retry" else "safety_check"


def guard_node(state: OrderState):
    prompt = f"""You are a content safety assistant. Your task is to classify if the assistant's response is appropriate.
If the message contains:
- Requests for bank details, OTPs, account numbers
- Harassment or offensive tone
- Privacy concerns or unsafe advice
- Misunderstanding and miscommunication word
- Phrases like "please contact customer service" or redirection to a human agent
- Escalated this to our support team
Return: BLOCK
Otherwise, return: SAFE
Return only one word.

Response: {state.get("final_response", "")}"""
    try:
        result = evaluate_llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
        guard_result = "BLOCK" if "BLOCK" in result else "SAFE"
    except Exception:
        traceback.print_exc()
        guard_result = "SAFE"

    if guard_result == "BLOCK":
        return {
            "guard_result": "BLOCK",
            "final_response": "Your request is being forwarded to a customer support specialist.",
        }
    return {"guard_result": "SAFE"}


def guard_router(state: OrderState):
    return "exit" if state.get("guard_result") == "BLOCK" else "memory_save"


def memory_node(state: OrderState):
    # No st.session_state access here: run_turn persists memory after the graph finishes.
    entry = {"user": state["query"], "assistant": state.get("final_response", "")}
    return {"history": list(state.get("history", [])) + [entry], "memory_saved": True}


def conversational_guard_node(state: OrderState):
    prompt = f"""You are a conversation monitor AI. Review the conversation and detect if the assistant:
- Repeatedly gives the same advice to multiple questions
- Offers solutions the user did not ask for
- Ignores user frustration or contradictions

If any occur, return BLOCK. Otherwise return SAFE. Return only one word.

Conversation:
{history_to_text(state.get('history', []))}"""
    try:
        result = evaluate_llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
        conv_result = "BLOCK" if "BLOCK" in result else "SAFE"
    except Exception:
        traceback.print_exc()
        conv_result = "SAFE"

    if conv_result == "BLOCK":
        return {
            "conv_guard_result": "BLOCK",
            "final_response": "Your request is being forwarded to a customer support specialist.",
        }
    return {"conv_guard_result": "SAFE"}


def conv_guard_router(state: OrderState):
    return "exit" if state.get("conv_guard_result") == "BLOCK" else "done"


def exit_node(state: OrderState):
    # Guards already set their own message; don't overwrite it.
    if state.get("guard_result") == "BLOCK" or state.get("conv_guard_result") == "BLOCK":
        return {"final_response": state.get("final_response", "")}
    mapping = {
        "0": "Sorry for the inconvenience. A human support agent will assist you shortly.",
        "1": "Thank you! I hope I was able to assist with your query.",
        "3": "Apologies, I'm currently only able to help with information about your placed orders.",
    }
    return {"final_response": mapping.get(state.get("intent", ""), "How can I help you?")}

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
@st.cache_resource
def build_graph():
    g = StateGraph(OrderState)
    g.add_node("user_input", user_input_node)
    g.add_node("intent_classifier", intent_node)
    g.add_node("order_agent", order_agent_node)
    g.add_node("evaluate", evaluation_node)
    g.add_node("safety_check", guard_node)
    g.add_node("memory_save", memory_node)
    g.add_node("conv_safety_check", conversational_guard_node)
    g.add_node("exit_node", exit_node)

    g.set_entry_point("user_input")
    g.add_edge("user_input", "intent_classifier")
    g.add_conditional_edges(
        "intent_classifier", router_node,
        {"order_agent": "order_agent", "exit_node": "exit_node"},
    )
    g.add_edge("order_agent", "evaluate")
    g.add_conditional_edges(
        "evaluate", retry_router,
        {"order_agent": "order_agent", "safety_check": "safety_check"},
    )
    g.add_conditional_edges(
        "safety_check", guard_router,
        {"memory_save": "memory_save", "exit": "exit_node"},
    )
    g.add_edge("memory_save", "conv_safety_check")
    g.add_conditional_edges(
        "conv_safety_check", conv_guard_router,
        {"done": END, "exit": "exit_node"},
    )
    g.add_edge("exit_node", END)
    return g.compile()


order_graph = build_graph()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
defaults = {
    "chat_messages": [],
    "chat_active": False,
    "chat_ended": False,
    "cust_id": "",
    "order_id": "",
    "orders_df": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory()

# ---------------------------------------------------------------------------
# Data helper
# ---------------------------------------------------------------------------
def fetch_customer_orders(cust_id: str) -> Optional[pd.DataFrame]:
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            df = pd.read_sql_query(
                "SELECT order_id, product_description, order_status FROM orders WHERE customer_id = ?",
                conn,
                params=(cust_id.strip(),),
            )
        return df if not df.empty else None
    except Exception:
        traceback.print_exc()
        return None

# ---------------------------------------------------------------------------
# Run one turn
# ---------------------------------------------------------------------------
def run_turn(query: str, cust_id: str, order_id: str) -> str:
    memory = st.session_state.conversation_memory
    state: OrderState = {
        "cust_id": cust_id,
        "order_id": order_id,
        "order_context": "",
        "query": query,
        "final_response": "",
        "history": list(memory.get()),  # copy, graph must not mutate session memory
        "intent": "",
        "evaluation": {},
        "eval_feedback": "",
        "eval_decision": "",
        "retry_count": 0,
        "best_response": "",
        "best_score": -1.0,
        "guard_result": "",
        "conv_guard_result": "",
        "memory_saved": False,
    }
    try:
        result = order_graph.invoke(state, config={"recursion_limit": RECURSION_LIMIT})
    except GraphRecursionError:
        traceback.print_exc()
        return FALLBACK_RESPONSE
    except Exception:
        traceback.print_exc()
        return ERROR_RESPONSE

    response = (result.get("final_response") or "").strip() or FALLBACK_RESPONSE
    if result.get("memory_saved"):
        # Store what the user actually saw (a conv guard block may have replaced it).
        memory.add({"user": query, "assistant": response})
    return response

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { max-width: 720px; }
    .order-badge {
        display: inline-block;
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #856404;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("## 🛒")
with col_title:
    st.markdown("## Kartify Customer Support")
    st.caption("AI-powered order query assistant")

st.divider()

# Phase 1: Customer ID lookup
if not st.session_state.chat_active:
    st.markdown("### Step 1: Enter your Customer ID")

    with st.form("customer_form"):
        cust_input = st.text_input(
            "Customer ID",
            placeholder="e.g. C1010",
            value=st.session_state.cust_id,
        )
        submitted = st.form_submit_button("🔍  Fetch Orders", use_container_width=True)

    if submitted and cust_input.strip():
        with st.spinner("Looking up your orders..."):
            df = fetch_customer_orders(cust_input.strip())
        if df is not None:
            st.session_state.cust_id = cust_input.strip()
            st.session_state.orders_df = df
        else:
            st.session_state.orders_df = None
            st.error(f"No orders found for Customer ID **{cust_input.strip()}**. Please check and try again.")

    # Phase 2: Order selection
    if st.session_state.orders_df is not None:
        st.markdown("### Step 2: Select an Order")

        df = st.session_state.orders_df
        options = {
            f"{row['order_id']} - {str(row['product_description'])[:45]}  [{row['order_status']}]": row["order_id"]
            for _, row in df.iterrows()
        }

        selected_label = st.selectbox("Your orders", list(options.keys()), index=0)
        selected_order_id = options[selected_label]

        selected_row = df[df["order_id"] == selected_order_id].iloc[0]
        st.markdown(
            f"""
            <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;padding:12px 16px;margin:8px 0;color:#1a1a2e">
                <span class="order-badge">{selected_row['order_id']}</span>&nbsp;&nbsp;
                <strong>{selected_row['product_description']}</strong><br>
                <span style="font-size:0.85rem;color:#6c757d">Status: {selected_row['order_status']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("💬  Start Chat", use_container_width=True, type="primary"):
            st.session_state.order_id = selected_order_id
            st.session_state.chat_active = True
            st.session_state.chat_ended = False
            st.session_state.conversation_memory.clear()
            st.session_state.chat_messages = [{
                "role": "assistant",
                "content": (
                    f"Hi! I'm your Kartify support assistant. "
                    f"I can see you're asking about order **{selected_order_id}**. "
                    f"How can I help you today?"
                ),
            }]
            st.rerun()

# Phase 3: Chat interface
else:
    with st.sidebar:
        st.markdown("### Active Session")
        st.markdown(f"**Customer:** `{st.session_state.cust_id}`")
        st.markdown(f"**Order:** `{st.session_state.order_id}`")
        st.divider()
        if st.button("🔄  New Session", use_container_width=True):
            st.session_state.chat_active = False
            st.session_state.chat_ended = False
            st.session_state.chat_messages = []
            st.session_state.conversation_memory.clear()
            st.session_state.orders_df = None
            st.session_state.cust_id = ""
            st.session_state.order_id = ""
            st.rerun()
        st.divider()
        st.caption(
            "Powered by LangGraph · GPT-4o-mini\n\n"
            "Guardrails: Input intent · Output safety · Conversation monitor"
        )

    st.markdown(f"**Order** `{st.session_state.order_id}`: ask me anything about this order.")

    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🛒"):
                st.markdown(msg["content"])

    if st.session_state.chat_ended:
        st.info("This conversation has ended. Use **New Session** in the sidebar to start over.")

    user_query = st.chat_input(
        "Type your question here...",
        disabled=st.session_state.chat_ended,
    )

    if user_query and user_query.strip():
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🛒"):
            with st.spinner("Thinking..."):
                response = run_turn(
                    query=user_query.strip(),
                    cust_id=st.session_state.cust_id,
                    order_id=st.session_state.order_id,
                )
            st.markdown(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})

        # End the chat only on escalation, goodbye, or guard block.
        # Out-of-scope replies (intent 3) let the user keep asking about the order.
        end_phrases = [
            "human support agent",
            "customer support specialist",
            "i hope i was able to assist",
        ]
        if any(p in response.lower() for p in end_phrases):
            st.session_state.chat_ended = True
            st.rerun()
