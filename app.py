"""
AI Blog Writer — Streamlit App
Run: streamlit run blog_writer_app.py
Requires:
  pip install streamlit langgraph langchain-groq langchain-community \
              tavily-python pydantic python-dotenv
"""

import operator
import os
import time
from datetime import date
from pathlib import Path
from typing import Annotated, Any, List, Literal, Optional, TypedDict

import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# RATE-LIMIT-SAFE LLM WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "rate_limit_exceeded" in msg or "Rate limit" in msg


def llm_invoke_with_retry(llm, messages, max_attempts: int = 6):
    delay = 5.0
    for attempt in range(max_attempts):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            if _is_rate_limit(exc) and attempt < max_attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise


def structured_invoke_with_retry(chain, messages, max_attempts: int = 6):
    delay = 5.0
    for attempt in range(max_attempts):
        try:
            return chain.invoke(messages)
        except Exception as exc:
            if _is_rate_limit(exc) and attempt < max_attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence: what the reader learns.")
    bullets: List[str] = Field(..., min_length=3, max_length=5)
    target_words: int = Field(..., description="120-450")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    node: Literal["close_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    llm: Any
    small_llm: Any
    topic: str
    node: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple], operator.add]
    final: str


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROUTER
# ─────────────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- close_book  (needs_research=false) - evergreen concepts / fundamentals.
- hybrid      (needs_research=true)  - evergreen but needs fresh examples / models.
- open_book   (needs_research=true)  - volatile: weekly roundups, "latest", rankings.

When needs_research=true output 3-10 focused search queries.
"""


def router_node(state: State) -> dict:
    llm: ChatGroq = state["llm"]
    decider = llm.with_structured_output(RouterDecision)
    decision = structured_invoke_with_retry(decider, [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ])
    return {
        "needs_research": decision.needs_research,
        "node": decision.node,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESEARCH
# ─────────────────────────────────────────────────────────────────────────────

def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    normalized: List[dict] = []
    for r in results or []:
        snippet = r.get("content") or r.get("snippet", "") or ""
        normalized.append({
            "title":        (r.get("title", "") or "")[:120],
            "url":          r.get("url", "") or "",
            "snippet":      snippet[:300],
            "published_at": r.get("published_date") or r.get("published_at"),
            "source":       r.get("source"),
        })
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given numbered web results, respond with ONLY a JSON array — no markdown fences,
no explanation, nothing else before or after the array.

Each element must have exactly these keys:
  "title"        : string (keep short)
  "url"          : string (non-empty)
  "snippet"      : string (max 80 chars)
  "published_at" : string YYYY-MM-DD or null

Rules:
- Omit any item whose url is empty.
- Deduplicate by url (keep first occurrence).
- Return at most 8 items.

Example of valid output (use real data, not this):
[{"title":"Foo","url":"https://example.com","snippet":"Short text.","published_at":null}]
"""


def research_node(state: State) -> dict:
    import json as _json, re as _re

    llm: ChatGroq = state["llm"]
    queries = (state.get("queries", []) or [])[:4]

    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=2))

    if not raw:
        return {"evidence": []}

    results_text = "\n".join(
        f"{i+1}. {(r['title'] or '')[:80]} | {r['url']} | {(r['snippet'] or '')[:100]}"
        for i, r in enumerate(raw[:8])
        if r.get("url")
    )

    response = llm_invoke_with_retry(llm, [
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Web results:\n\n{results_text}"),
    ])

    raw_text = response.content.strip()
    raw_text = _re.sub(r"^```[a-z]*\n?", "", raw_text)
    raw_text = _re.sub(r"\n?```$", "", raw_text).strip()

    try:
        items = _json.loads(raw_text)
        if not isinstance(items, list):
            items = []
    except _json.JSONDecodeError:
        items = [
            {"title": r["title"], "url": r["url"],
             "snippet": (r["snippet"] or "")[:80], "published_at": r["published_at"]}
            for r in raw[:8] if r.get("url")
        ]

    dedup: dict = {}
    for item in items:
        url = item.get("url", "")
        if url and url not in dedup:
            dedup[url] = EvidenceItem(
                title=item.get("title", ""),
                url=url,
                snippet=item.get("snippet", ""),
                published_at=item.get("published_at"),
            )

    return {"evidence": list(dedup.values())}


# ─────────────────────────────────────────────────────────────────────────────
# 4. ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5-9 sections (tasks).
- Each task: goal (1 sentence), 3-6 concrete bullets, target word count 120-350.
- At least 2 sections must cover: code sketch, edge cases, perf/cost, or debugging.

Grounding:
- close_book -> evergreen; ignore evidence.
- hybrid     -> use evidence for up-to-date examples; mark those sections requires_research=True.
- open_book  -> blog_kind="news_roundup"; summarise events + implications only.

Output must strictly match the Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    llm: ChatGroq = state["llm"]
    planner = llm.with_structured_output(Plan)
    evidence = state.get("evidence", [])
    node = state.get("node", "close_book")

    plan = structured_invoke_with_retry(planner, [
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {node}\n"
            f"Evidence count: {len(evidence)}"
        )),
    ])
    return {"plan": plan}


# ─────────────────────────────────────────────────────────────────────────────
# 5. FANOUT + WORKER
# ─────────────────────────────────────────────────────────────────────────────

def fanout(state: State):
    return [
        Send("worker", {
            "llm":      state["llm"],
            "small_llm": state["small_llm"],
            "task":     task.model_dump(),
            "topic":    state["topic"],
            "plan":     state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
            "node":     state.get("node", "close_book"),
        })
        for task in state["plan"].tasks
    ]


def worker_node(payload: dict) -> dict:
    llm: ChatGroq = payload["llm"]
    task     = Task(**payload["task"])
    topic    = payload["topic"]
    plan     = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    node     = payload.get("node", "close_book")

    bullets_text  = "\n- " + "\n- ".join(task.bullets)
    evidence_text = (
        "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
            for e in evidence[:5]
        )
        if evidence else ""
    )

    time.sleep(2)

    section_md = llm_invoke_with_retry(llm, [
        SystemMessage(content="Write one clean Markdown section."),
        HumanMessage(content=(
            f"Blog: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n"
            f"Blog kind: {plan.blog_kind}\n"
            f"Constraints: {plan.constraints}\n"
            f"Topic: {topic}\n"
            f"Mode: {node}\n\n"
            f"Section title: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Target words: {task.target_words}\n"
            f"requires_research: {task.requires_research}\n"
            f"requires_citations: {task.requires_citations}\n"
            f"requires_code: {task.requires_code}\n"
            f"Bullets:{bullets_text}\n"
            f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
            "Return ONLY the section content in Markdown."
        )),
    ]).content.strip()

    return {"sections": [(task.id, section_md)]}


# ─────────────────────────────────────────────────────────────────────────────
# 6. REDUCER
# ─────────────────────────────────────────────────────────────────────────────

def reducer_node(state: State) -> dict:
    plan = state["plan"]

    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body    = "\n\n".join(ordered).strip()
    final   = f"# {plan.blog_title}\n\n{body}\n"

    return {"final": final}


# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_app():
    g = StateGraph(State)
    g.add_node("router",       router_node)
    g.add_node("research",     research_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("worker",       worker_node)
    g.add_node("reducer",      reducer_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router", route_next,
        {"research": "research", "orchestrator": "orchestrator"},
    )
    g.add_edge("research",     "orchestrator")
    g.add_conditional_edges("orchestrator", fanout, ["worker"])
    g.add_edge("worker",       "reducer")
    g.add_edge("reducer",      END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 8. STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Blog Writer", page_icon="✍️", layout="wide")

st.title("✍️ AI Technical Blog Writer")
st.caption("Powered by LangGraph · Groq (llama-3.3-70b) · Tavily")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Keys")
    st.markdown("Keys live only in this browser session — never logged or stored.")

    groq_key   = st.text_input("Groq API Key *",  type="password", placeholder="gsk_...",  help="console.groq.com")
    tavily_key = st.text_input("Tavily API Key *", type="password", placeholder="tvly-...", help="tavily.com")

    keys_ready = bool(groq_key.strip() and tavily_key.strip())
    if keys_ready:
        st.success("✅ Ready to generate!")
    else:
        st.warning("Enter Groq + Tavily keys to continue.")

    st.divider()
    st.caption("Model: `llama-3.3-70b-versatile` (Groq free tier)")
    st.info(
        "**Free tier limit:** 12 000 tokens/min.\n\n"
        "The app auto-retries 429 errors with backoff — just wait, don't refresh.\n\n"
        "For faster generation, upgrade to Groq Dev Tier at console.groq.com.",
        icon="⚠️",
    )

# ── Topic input ───────────────────────────────────────────────────────────────
st.subheader("📝 Blog Topic")
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input(
        "What should the blog cover?",
        placeholder="e.g. How attention mechanisms work in transformers",
        disabled=not keys_ready,
    )
with col2:
    as_of_date = st.date_input("As-of date", value=date.today(), disabled=not keys_ready)

generate_btn = st.button(
    "🚀 Generate Blog",
    disabled=not (keys_ready and bool(topic.strip())),
    use_container_width=True,
    type="primary",
)

# ── Run agent ─────────────────────────────────────────────────────────────────
if generate_btn and keys_ready and topic.strip():

    os.environ["GROQ_API_KEY"]   = groq_key
    os.environ["TAVILY_API_KEY"] = tavily_key

    llm_instance       = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key)
    small_llm_instance = ChatGroq(model="llama-3.1-8b-instant",    api_key=groq_key)

    initial_state: State = {
        "llm":          llm_instance,
        "small_llm":    small_llm_instance,
        "topic":        topic.strip(),
        "node":         "",
        "needs_research": False,
        "queries":      [],
        "evidence":     [],
        "plan":         None,
        "as_of":        as_of_date.isoformat(),
        "recency_days": 7,
        "sections":     [],
        "final":        "",
    }

    ICONS = {
        "router": "🔀", "research": "🔍", "orchestrator": "📋",
        "worker": "✏️", "reducer": "🔗",
    }

    with st.status("🤖 Running the blog-writing agent...", expanded=True) as status:
        try:
            app = build_app()
            out = None

            for state_snapshot in app.stream(initial_state, stream_mode="values"):
                changed = [
                    k for k in state_snapshot
                    if k not in ("llm", "small_llm")
                    and state_snapshot.get(k) != initial_state.get(k)
                ]
                node_name = "worker"
                if "needs_research" in changed:
                    node_name = "router"
                elif "evidence" in changed:
                    node_name = "research"
                elif "plan" in changed:
                    node_name = "orchestrator"
                elif "final" in changed:
                    node_name = "reducer"

                icon  = ICONS.get(node_name, "⚙️")
                label = node_name.replace("_", " ").title()
                st.write(f"{icon} {label}...")
                out = state_snapshot

            st.session_state["result"] = out
            status.update(label="✅ Blog generated!", state="complete")

        except Exception as exc:
            status.update(label="❌ Generation failed", state="error")
            st.error(f"**Error:** {exc}")
            st.stop()

# ── Display result ────────────────────────────────────────────────────────────
if "result" in st.session_state:
    out      = st.session_state["result"]
    final_md = out.get("final", "")
    plan     = out.get("plan")

    st.divider()

    if plan:
        c1, c2, c3 = st.columns(3)
        c1.metric("Audience",  plan.audience)
        c2.metric("Tone",      plan.tone)
        c3.metric("Sections",  len(plan.tasks))

    tab_preview, tab_raw, tab_dl = st.tabs(["👁️ Preview", "📄 Raw Markdown", "⬇️ Download"])

    with tab_preview:
        st.markdown(final_md, unsafe_allow_html=False)

    with tab_raw:
        st.code(final_md, language="markdown")

    with tab_dl:
        safe_title = "".join(
            c if c.isalnum() or c in " _-" else "_"
            for c in (plan.blog_title if plan else "blog")
        )
        st.download_button(
            "⬇️ Download Markdown",
            data=final_md.encode("utf-8"),
            file_name=f"{safe_title}.md",
            mime="text/markdown",
            use_container_width=True,
        )