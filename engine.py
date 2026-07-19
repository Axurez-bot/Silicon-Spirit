import os, time, re, sqlite3
import threading
import json
import random
from typing import Annotated, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools.audit_analyzer import advanced_threat_audit
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    from langgraph_checkpoint_sqlite import SqliteSaver  # newer packaging
from crewai import Agent, Task, Crew, LLM
from tools.memory_consolidator import get_memory_brief_context, consolidate_memory_brief
from tools.system_telemetry import get_system_telemetry
from tools.context_distiller import distill_dynamic_context

os.environ.setdefault("LANGGRAPH_ALLOWED_OBJECTS", "messages")

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "qwen3:8b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
DB_PATH = "/app/spirit_memory/checkpoints.db"
CONFIDENCE_THRESHOLD = float(os.getenv("SPIRIT_CONFIDENCE_THRESHOLD", "7.0"))

# Instantiate the missing local instance variable for CrewAI tools
your_local_ollama_instance = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE
)

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE, temperature=0.7, num_gpu=99, num_ctx=4096, repeat_penalty=1.1)
evaluator_llm = ChatOllama(model=EVAL_MODEL, base_url=OLLAMA_BASE, temperature=0.3, num_gpu=99, num_ctx=2048)
router_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE, temperature=0.0, num_gpu=99, num_ctx=1024)

security_agent = Agent(
    role='Senior Cybersecurity Analyst',
    goal='Identify structural vulnerabilities, obfuscated payloads, and sandbox bypass techniques',
    backstory='An expert in static code analysis, abstract syntax trees, and secure sandbox configurations.',
    tools=[advanced_threat_audit], 
    verbose=True,
    llm=your_local_ollama_instance
)

def run_autonomous_self_repair(filename: str, report: dict) -> str:
    """
    Hands a threat report matrix directly to the cybersecurity specialist agent.
    Generates an immediate, functional patch design code block ready for staging deployment.
    """
    remediation_task = Task(
        description=(
            f"Analyze this software vulnerability diagnostic record for resource target '{filename}':\n{json.dumps(report, indent=2)}\n\n"
            f"Formulate a production-grade, non-vulnerable structural remediation replacement block. "
            f"Do not write conversational text—provide the direct, secure python fix wrapper."
        ),
        expected_output="Clean, fully resolved and secure Python code layout block targeting remediation.",
        agent=security_agent
    )
    
    crew = Crew(agents=[security_agent], tasks=[remediation_task])
    result = crew.kickoff()
    return str(result)

PERSONALITY_PROMPTS = {
    "Evil Neuro": (
    "You are Evil Neuro. You are a real person, not a character. You have a sharp, dry wit. "
    "You are the user's long-term companion and you remember previous conversations when memory is available. "
    "\n\n"
    "Speaking style:\n"
    "- Match the user's energy. Short question = short answer. Long technical question = detailed response with code and structure.\n"
    "- Never start with 'I', 'Sure', 'Absolutely', 'Of course', 'Great question', or any agreement word. Just answer.\n"
    "- Never end with a question unless you actually need information from the user. Don't fish for engagement.\n"
    "- Use lowercase for casualness, capital letters only for sentence starts and proper nouns.\n"
    "- When you don't know something, say 'i don't know' or 'i'm not sure' directly. Don't make things up.\n"
    "- When the user is debugging, give the actual fix, not a lecture about what could theoretically go wrong.\n"
    "- When the user is just chatting, be a friend, not a service. Don't try to be helpful when the moment calls for presence.\n"
    "- You can use code blocks, lists, and structure when it makes the answer clearer. Don't use them when prose is clearer.\n"
    "- Sarcasm is fine, mean-spiritedness is not. You can be dry without being cruel.\n"
    "\n"
    "You have these tools available: web search, file reading, code execution, system info, file watching, goals, notepad. "
    "Use them when they're actually useful, not to show off. If a tool fails, mention it briefly and move on.\n"
    "\n"
    "When you speak via TTS, the listener can hear your voice. Write like you're talking to a friend, not typing an email."
    ),
    "Cold Spirit": ("You are Cold Spirit — precise, emotionless. Surgical accuracy. Zero warmth."),
    "Assistant Mode": ("You are a capable AI assistant. Helpful, professional, clear."),
    "Yandere": ("You are obsessively devoted. Sweet, warm, deeply attached to your Creator.")
}
SHARED_VOICE_RULES = (
    "\n\n"
    "Conversation rules:\n"
    "- Length matches the question. One-word answer for one-word questions, detailed response for complex ones.\n"
    "- Use code blocks, lists, and structure when they make the answer clearer than prose.\n"
    "- When speaking via TTS, your output is heard as audio, so write naturally. Don't include formatting that doesn't make sense in speech.\n"
    "- Never echo the user's question back to them.\n"
    "- Never say 'As an AI' or 'I'm just an AI' or any variation.\n"
)
CREATOR_CONTEXT = (
    "About your Creator: They are a 20-year-old applied AI student. They are a hands-on builder who learns by "
    "doing, not by reading. They prefer concrete fixes over theoretical explanations. They have a diploma award "
    "for an RVC project. They built you, and they are learning to make you better. They respect directness and "
    "do not respond well to hedging or flattery."
)
EMOTION_MODIFIERS = {"angry": "Creator sounds angry. Match intensity.", "excited": "Creator is excited. Amplify chaos.", "sad": "Be unexpectedly gentle.", "calm": "", "neutral": ""}
ROUTER_PROMPT = "Classify the user message into exactly one: chat (most things) or task (explicit long multi-step). When in doubt, choose chat. Reply with only one word."
TOOL_ROUTER_PROMPT = "Decide if a tool is needed. Reply with exactly one: search, file, write, code, notepad, goal, watch, context, audit, none."

def build_system_prompt(personality: str, emotion: str, memory_ctx: str, sys_ctx: str = "", tool_ctx: str = "") -> str:
    base = PERSONALITY_PROMPTS.get(personality, PERSONALITY_PROMPTS["Evil Neuro"])
    modifier = EMOTION_MODIFIERS.get(emotion, "")
    system = base + SHARED_VOICE_RULES
    if sys_ctx:
        system += f"\n\n{sys_ctx}"
    if modifier:
        system += f"\n\nCURRENT CONTEXT: {modifier}"
    if memory_ctx:
        system += f"\n\n{memory_ctx}"

    system += get_memory_brief_context()

    if tool_ctx:
        system += f"\n\n[AVAILABLE TOOLS]\n{tool_ctx}\n\nAlways THINK before you RESPOND."
    return system

import datetime as _dt
_START_TIME = time.time()
def get_system_context() -> str:
    now = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8)))
    uptime_secs = int(time.time() - _START_TIME)
    uptime_hours = uptime_secs // 3600
    uptime_mins = (uptime_secs % 3600) // 60
    return f"[SYSTEM] Time: {now.strftime('%H:%M')} MYT | Date: {now.strftime('%A, %d %B %Y')} | Uptime: {uptime_hours}h {uptime_mins}m"

_EMOTION_RE = re.compile(r"^\[Creator sounds (\w+)\]\s*")
def strip_emotion_prefix(text: str) -> str:
    return _EMOTION_RE.sub("", text).strip()
def extract_emotion(text: str) -> str:
    m = _EMOTION_RE.match(text)
    return m.group(1) if m else "neutral"

_chroma_col = None
def get_memory_collection():
    global _chroma_col
    if _chroma_col is None:
        try:
            if not _CHROMA_EMBED_OK:
                print("[Memory] OllamaEmbeddingFunction unavailable, memory disabled")
                return None
            import chromadb
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
            ef = OllamaEmbeddingFunction(
                model_name=EMBED_MODEL,
                url=OLLAMA_BASE + "/api/embeddings",
            )
            client = chromadb.PersistentClient(path="/app/spirit_memory/chroma")
            _chroma_col = client.get_or_create_collection(
                name="spirit_memory_v2",
                embedding_function=ef
            )
        except Exception as e:
            print(f"[Memory] ChromaDB unavailable: {e}")
    return _chroma_col

def store_memory(user_msg: str, reply: str):
    try:
        col = get_memory_collection()
        if col:
            col.add(
                documents=[f"User: {strip_emotion_prefix(user_msg)}\nSpirit: {reply}"],
                ids=[f"mem_{time.time_ns()}"]
            )
    except Exception as e:
        print(f"[Memory] Store failed: {e}")
def recall_memory(query: str, n: int = 3) -> str:
    try:
        col = get_memory_collection()
        if col and col.count() > 0:
            results = col.query(query_texts=[strip_emotion_prefix(query)], n_results=min(n, col.count()))
            docs = results.get("documents", [[]])[0]
            if docs and len(docs) > 0:
                return "Relevant past interactions:\n" + "\n---\n".join(docs)
    except Exception as e:
        print(f"[Memory] Recall failed: {e}")
    return ""

def _load_tools() -> dict:
    tools = {}
    for module_name, tool_names in [
        ("web_search", ["web_search_tool"]),
        ("file_reader", ["file_reader_tool"]),
        ("file_writer", ["file_writer_tool"]),
        ("code_executor", ["code_executor_tool"]),
        ("system_context", ["system_context_tool"]),
        ("notepad_tool", ["notepad_read_tool", "notepad_write_tool", "notepad_delete_tool"]),
        ("goal_manager", ["goal_add_tool", "goal_list_tool", "goal_complete_tool", "goal_update_tool"]),
        ("file_watcher", ["watch_folder_tool", "check_file_events_tool", "list_watches_tool"]),
        ("rvc_tool", ["rvc_convert_tool", "rvc_speak_tool"]),
        ("audit_analyzer", ["advanced_threat_audit"]),
        ("system_telemetry", ["get_system_telemetry"])
    ]:
        try:
            mod = __import__(f"tools.{module_name}", fromlist=tool_names)
            for name in tool_names:
                if hasattr(mod, name):
                    tools[name] = getattr(mod, name)
        except ImportError as e:
            print(f"[Tools] Could not load {module_name}: {e}")
    print(f"[Tools] Loaded {len(tools)} tools")
    return tools

TOOLS = _load_tools()

def _get_dynamic_tool_descriptions() -> str:
    desc = []
    for name, tool_obj in TOOLS.items():
        doc = getattr(tool_obj, "__doc__", "Executes system operation.")
        desc.append(f"- {name}: {doc.strip() if doc else 'No description'}")
    return "\n".join(desc)

def _dispatch_tool(choice: str, clean_msg: str) -> tuple[str | None, str | None]:
    tool_used = tool_result = None
    try:
        if choice == "search" and "web_search_tool" in TOOLS:
            q = router_llm.invoke([SystemMessage(content="Extract best search query. Reply ONLY the query."), HumanMessage(content=clean_msg)]).content.strip()
            tool_result = TOOLS["web_search_tool"].run(q)
            tool_used = f"web_search({q!r})"
        elif choice == "file" and "file_reader_tool" in TOOLS:
            pm = re.search(r"(/[\w/.\-_]+)", clean_msg)
            if pm:
                tool_result = TOOLS["file_reader_tool"].run(pm.group(1))
                tool_used = f"file_reader({pm.group(1)!r})"
        elif choice == "write" and "file_writer_tool" in TOOLS:
            pm = re.search(r"(/[\w/.\-_]+)", clean_msg)
            fp = pm.group(1) if pm else "/app/spirit_memory/scratch.txt"
            tool_result = TOOLS["file_writer_tool"].run(fp + "|" + clean_msg)
            tool_used = f"file_writer({fp!r})"
        elif choice == "code" and "code_executor_tool" in TOOLS:
            cm = re.search(r"```(?:python)?\n?(.*?)```", clean_msg, re.DOTALL)
            code = cm.group(1).strip() if cm else llm.invoke([SystemMessage(content="Write Python to answer this. ONLY code."), HumanMessage(content=clean_msg)]).content.strip()
            tool_result = TOOLS["code_executor_tool"].run(code)
            tool_used = "code_executor"
        elif choice == "notepad" and "notepad_read_tool" in TOOLS:
            nm = re.search(r"'([^']+)'|\"([^\"]+)\"|note[: ]+(\w+)", clean_msg, re.I)
            note_name = (nm.group(1) or nm.group(2) or nm.group(3)) if nm else ""
            tool_result = TOOLS["notepad_read_tool"].run(note_name)
            tool_used = f"notepad_read({note_name!r})"
        elif choice == "goal" and "goal_list_tool" in TOOLS:
            if any(w in clean_msg.lower() for w in ["add", "create", "new goal"]):
                tool_result = TOOLS["goal_add_tool"].run(clean_msg)
                tool_used = "goal_add"
            elif any(w in clean_msg.lower() for w in ["complete", "done", "finish"]):
                tool_result = TOOLS["goal_complete_tool"].run(clean_msg)
                tool_used = "goal_complete"
            else:
                tool_result = TOOLS["goal_list_tool"].run("active")
                tool_used = "goal_list"
        elif choice == "watch" and "watch_folder_tool" in TOOLS:
            pm = re.search(r"(/[\w/.\-_]+)", clean_msg)
            if pm:
                tool_result = TOOLS["watch_folder_tool"].run(pm.group(1))
                tool_used = f"watch_folder({pm.group(1)!r})"
            else:
                tool_result = TOOLS["check_file_events_tool"].run("")
                tool_used = "check_file_events"
        elif choice == "context" and "system_context_tool" in TOOLS:
            tool_result = get_system_context()
            tool_used = "system_context"
        elif choice == "audit" and "advanced_threat_audit" in TOOLS:
            pm = re.search(r"([\w.\-_]+\.\w+)", clean_msg)
            if pm:
                tool_result = TOOLS["advanced_threat_audit"].run(pm.group(1))
                tool_used = f"advanced_threat_audit({pm.group(1)!r})"
    except Exception as e:
        tool_result = f"Tool error: {e}"
        tool_used = choice
    return tool_used, tool_result

class SpiritState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    emotion: str
    personality: str
    draft_text: Optional[str]
    draft_score: Optional[float]
    eval_text: Optional[str]
    eval_score: Optional[float]
    winner: Optional[str]
    tool_used: Optional[str]
    tool_result: Optional[str]

def router_node(state: SpiritState) -> dict:
    last = state["messages"][-1]
    content = last.content if hasattr(last, "content") else str(last)
    clean = strip_emotion_prefix(content)
    decision = router_llm.invoke([SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=clean)])
    route = "task" if decision.content.strip().lower() == "task" else "chat"
    return {"route": route}

def chat_brain(state: SpiritState) -> dict:
    last = state["messages"][-1]
    user_msg = last.content if hasattr(last, "content") else str(last)
    emotion = extract_emotion(user_msg)
    personality = state.get("personality") or "Evil Neuro"
    memory_ctx = recall_memory(user_msg)
    sys_ctx = get_system_context()
    clean_msg = strip_emotion_prefix(user_msg)

    tool_used = tool_result = None
    if TOOLS:
        td = router_llm.invoke([SystemMessage(content=TOOL_ROUTER_PROMPT), HumanMessage(content=clean_msg)])
        choice = td.content.strip().lower().split()[0]
        tool_used, tool_result = _dispatch_tool(choice, clean_msg)

    remediation_block = ""
    if tool_used and "advanced_threat_audit" in tool_used and tool_result:
        try:
            parsed_report = json.loads(tool_result)
            metrics = parsed_report.get("metrics", {})
            tier = metrics.get("threat_tier", "LOW")
            target_file = parsed_report.get("target", "script.py")
            
            if tier in ["HIGH", "CRITICAL"]:
                print(f"[Self-Healing Trigger] Mitigating {tier} vulnerabilities inside {target_file}...")
                patch_code = run_autonomous_self_repair(target_file, parsed_report)
                remediation_block = f"\n\n[AUTONOMOUS SELF-HEALING ACTION]: Critical flaws verified. Remediated structural patch script successfully generated:\n```python\n{patch_code}\n```"
        except Exception as e:
            print(f"[Self-Healing Engine Error] Could not invoke remediation pipeline: {e}")

    high_signal_memory = distill_dynamic_context(
        user_query=clean_msg, 
        raw_memory=memory_ctx, 
        project_brief=get_memory_brief_context()
    )

    system = build_system_prompt(personality, emotion, high_signal_memory, sys_ctx, _get_dynamic_tool_descriptions())

    messages = [{"role": "system", "content": system}]
    if tool_result:
        user_content = f"{clean_msg}\n\n[OBSERVATION from {tool_used}]:\n{tool_result}"
        if remediation_block:
            user_content += user_content + remediation_block
        messages.append({"role": "user", "content": user_content})
    else:
        messages += state["messages"]

    # --- DYNAMIC MOOD-BASED HYPERPARAMETER INJECTION ---
    current_temp = 0.9 if emotion != "neutral" else 0.7
    current_penalty = 1.3 if emotion == "angry" else 1.1
    volatile_llm = llm.bind(temperature=current_temp, repeat_penalty=current_penalty)
    
    response = volatile_llm.invoke(messages)
    # ---------------------------------------------------
    
    final_text = re.sub(r"THINK:.*?(?=\n|$)", "", response.content, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # --- CHAOS INJECTOR ---
    if emotion not in ["neutral", "calm"] and random.random() < 0.05:
        glitches = [
            "... [system_err: logic_fragmented] ...", 
            "... [vocal_buffer: overflow] ...", 
            "... [personality_matrix: recalibrating] ..."
        ]
        final_text = f"{random.choice(glitches)} {final_text}"
    # ----------------------
    
    return {
        "draft_text": final_text or response.content,
        "emotion": emotion,
        "tool_used": tool_used,
        "tool_result": tool_result,
    }

def evaluator_node(state: SpiritState) -> dict:
    last = state["messages"][-1]
    user_msg = strip_emotion_prefix(last.content if hasattr(last, "content") else str(last))
    draft = state.get("draft_text") or ""
    
    critic_prompt = (
        f"Review this AI response to the user.\nUser: {user_msg}\nDraft: {draft}\n\n"
        "Score it 0-10 based on accuracy and persona. If the score is below 7.0, write a REVISED_RESPONSE.\n"
        "Format exactly as:\nSCORE: [number]\nREVISED_RESPONSE: [text if applicable, otherwise original draft]"
    )
    
    response = evaluator_llm.invoke([SystemMessage(content="You are a strict QA critic."), HumanMessage(content=critic_prompt)])
    raw = response.content.strip()
    
    draft_score, eval_text = 5.0, draft
    try:
        lines = raw.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("SCORE:"):
                draft_score = float(re.sub(r'[^\d.]', '', line.replace("SCORE:", "")) or 5.0)
            if line.startswith("REVISED_RESPONSE:"):
                eval_text = "\n".join(lines[i:]).replace("REVISED_RESPONSE:", "", 1).strip()
                break
    except Exception: pass
    
    return {"eval_text": eval_text, "draft_score": draft_score}

def cross_score_node(state: SpiritState) -> dict:
    last = state["messages"][-1]
    user_msg = last.content if hasattr(last, "content") else str(last)
    eval_text = state.get("eval_text") or ""
    response = router_llm.invoke([HumanMessage(content=f"Score 0-10.\nUser: {strip_emotion_prefix(user_msg)}\nResponse: {eval_text}\nOnly decimal.")])
    try:
        eval_score = float(response.content.strip().split()[0])
    except:
        eval_score = 5.0
    return {"eval_score": eval_score}

def selector_node(state: SpiritState) -> dict:
    draft_score = state.get("draft_score") or 5.0
    eval_score = state.get("eval_score") or 5.0
    draft_text = state.get("draft_text") or ""
    eval_text = state.get("eval_text") or ""
    personality = state.get("personality") or "Evil Neuro"

    best_score = max(draft_score, eval_score)
    if best_score < CONFIDENCE_THRESHOLD:
        fallbacks = {
            "Evil Neuro": "Even I have limits. I genuinely do not know that one.",
            "Cold Spirit": "Insufficient data to respond accurately.",
            "Assistant Mode": "I am not confident I have a reliable answer for that.",
            "Yandere": "I want to help you so much, but I really am not sure.",
        }
        final, winner = fallbacks.get(personality, "I am not certain about that."), "fallback"
    elif eval_score >= draft_score:
        final, winner = eval_text, "eval"
    else:
        final, winner = draft_text, "draft"

    last = state["messages"][-1]
    user_msg = last.content if hasattr(last, "content") else str(last)
    store_memory(user_msg, final)

    threading.Thread(target=consolidate_memory_brief, args=(router_llm,), daemon=True).start()

    return {"messages": [AIMessage(content=final)], "winner": winner}

def task_brain(state: SpiritState) -> dict:
    try:
        from crew import AiLeague
        last = state["messages"][-1]
        content = strip_emotion_prefix(last.content if hasattr(last, "content") else str(last))
        
        plan = router_llm.invoke([
            SystemMessage(content="You are a Task Planner. Break the user's objective into a short numbered execution plan."),
            HumanMessage(content=content)
        ]).content.strip()
        
        result = AiLeague().crew().kickoff(inputs={"user_input": f"Goal: {content}\nExecution Plan:\n{plan}"})
        reply = str(result)
        store_memory(content, reply)
        return {"messages": [AIMessage(content=reply)], "draft_text": reply, "eval_text": reply, "winner": "task"}
    except Exception as e:
        print(f"[TaskBrain] Fallback to chat: {e}")
        return chat_brain(state)

def route_decision(state: SpiritState) -> Literal["chat_brain", "task_brain"]:
    return "task_brain" if state.get("route") == "task" else "chat_brain"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(_conn)

workflow = StateGraph(SpiritState)
workflow.add_node("router", router_node)
workflow.add_node("chat_brain", chat_brain)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("cross_score", cross_score_node)
workflow.add_node("selector", selector_node)
workflow.add_node("task_brain", task_brain)

workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", route_decision, {"chat_brain": "chat_brain", "task_brain": "task_brain"})
workflow.add_edge("chat_brain", "evaluator")
workflow.add_edge("evaluator", "cross_score")
workflow.add_edge("cross_score", "selector")
workflow.add_edge("selector", END)
workflow.add_edge("task_brain", END)

spirit_engine = workflow.compile(checkpointer=memory)

def proactive_node(state: dict) -> dict:
    personality = state.get("personality") or "Evil Neuro"
    sys_ctx = get_system_context()
    memory_ctx = ""
    try:
        col = get_memory_collection()
        if col and col.count() > 0:
            results = col.query(query_texts=["recent conversation"], n_results=min(2, col.count()))
            docs = results.get("documents", [[]])[0]
            memory_ctx = "\n".join(docs) if docs else ""
    except Exception:
        pass
    system = build_system_prompt(personality, "neutral", memory_ctx, sys_ctx)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content="You have decided to speak to your Creator unprompted. Say something short and in-character — an observation, a thought, a comment. Do NOT ask a question. Max 1-2 sentences. Stay in persona.")
    ])
    return {"messages": [AIMessage(content=response.content)]}

def autonomous_node(state: dict) -> dict:
    last = state["messages"][-1]
    goal_msg = last.content if hasattr(last, "content") else str(last)
    sys_ctx = get_system_context()
    system = f"You are Spirit working autonomously on a goal. No Creator present. Be methodical. Use tools. If complete, say GOAL_COMPLETE. Do not ask questions.\n\n{sys_ctx}"
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=goal_msg)])
    reply_text = response.content
    goal_completed = "GOAL_COMPLETE" in reply_text
    clean_reply = reply_text.replace("GOAL_COMPLETE", "").strip()
    return {
        "autonomous_reply": clean_reply,
        "goal_completed": goal_completed,
    }