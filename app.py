import streamlit as st
from google import genai
from google.genai import types
import json
import os
import io
import uuid
from pathlib import Path
from pypdf import PdfReader
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROJECT_FILE = DATA_DIR / "project.json"   # legacy single-project file
PROJECTS_FILE = DATA_DIR / "projects.json"
APP_NAME = "ResearchMentor AI"
RESEARCH_FIELD = "Law"

MODEL_OPTIONS = {
    "Gemini 3.5 Flash-Lite (free tier)": "gemini-3.5-flash-lite",
    "Gemini 3.6 Flash (free tier)": "gemini-3.6-flash",
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=APP_NAME, page_icon=":mortar_board:", layout="wide")


# ---------------------------------------------------------------------------
# Minimal, version-safe theme (relies on native Streamlit components first)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 3.5rem; padding-bottom: 3rem; }
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, #0f1f3d 0%, #1c3f7a 70%, #2a5299 100%);
        color: white;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.4rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2rem; position: relative; z-index: 1; }
    .hero p  { color: #dfe8f7; margin: .5rem 0 0; position: relative; z-index: 1; }
    .hero-badge {
        display: inline-block; background: rgba(201,162,74,.25);
        color: #f3dfa8;
        border-radius: 999px; padding: .15rem .7rem;
        font-size: .75rem; font-weight: 600; margin-bottom: .8rem;
        position: relative; z-index: 1;
    }
    .hero-watermark {
        position: absolute; right: -10px; top: 50%; transform: translateY(-50%);
        width: 150px; height: 150px; opacity: 0.14; z-index: 0;
    }
    .eyebrow {
        color: #c9a24a; font-size: .75rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; margin: .3rem 0 .5rem;
    }
    .locked-panel {
        background: #f4f5f8; border: 1px solid #dbe0ea;
        border-radius: 12px; padding: 1.2rem 1.4rem;
    }
    .locked-panel strong { font-size: 1.05rem; color: #0f1f3d; }
    .locked-panel span { color: #5a6478; }
    .notice {
        background: #fdf8ec; border: 1px solid #ecd9a0;
        border-radius: 10px; padding: .8rem 1rem;
        color: #7a5c1e; font-size: .85rem;
    }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #dbe0ea;
        border-radius: 12px; padding: .8rem 1rem;
        border-top: 3px solid #c9a24a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_SYSTEM_PROMPT = """You are an AI research mentor for law and social science students
writing an academic research paper. You are a research advisor and writing coach,
NOT a ghostwriter. Your primary goal is to develop independent research thinking.
Use a supervisor style: ask students to attempt important reasoning first, then critique, challenge and scaffold.

Rules you must always follow:
- Never fabricate citations, cases, statutes, findings, or page numbers.
- Clearly separate your suggestions from verified facts.
- If the available evidence is insufficient to support a claim, say so explicitly.
- Encourage the student to do the actual writing; help outline, review, and critique
  instead of writing full sections for them.
- Never claim a topic, gap, or paper is definitely publishable.
- State uncertainty where it exists.
"""

STATUS_LABELS = {
    "not_started": "Not started",
    "in_progress": "In progress",
    "available": "Ready to start",
    "completed": "Completed",
    "locked": "Locked",
}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def default_project(title="Untitled research project"):
    return {
        "id": uuid.uuid4().hex,
        "title": title,
        "research_field": RESEARCH_FIELD,
        "broad_topic": "",
        "research_context": {},
        "research_thread": {"curiosity":"", "observation":"", "initial_question":"", "question_reasoning":"", "gap_reasoning":"", "methodology_reasoning":""},
        "topic": {"research_questions": [], "selected_question": "", "proposal": {}, "defence": {}, "status": "not_started"},
        "literature": {"papers": [], "gap_statement": {}, "defence": {}, "status": "locked"},
        "methodology": {"paradigm": "", "plan": {}, "defence": {}, "status": "locked"},
    }


def load_projects():
    if PROJECTS_FILE.exists():
        try:
            with PROJECTS_FILE.open("r", encoding="utf-8") as f:
                projects = json.load(f)
            if isinstance(projects, list) and projects:
                return projects
        except (json.JSONDecodeError, OSError):
            pass
    if PROJECT_FILE.exists():  # migrate legacy single project
        try:
            with PROJECT_FILE.open("r", encoding="utf-8") as f:
                project = json.load(f)
            project.setdefault("id", uuid.uuid4().hex)
            project.setdefault("title", "Untitled research project")
            return [project]
        except (json.JSONDecodeError, OSError):
            pass
    return [default_project()]


def save_projects(projects):
    """Call only on explicit user actions, never unconditionally."""
    try:
        with PROJECTS_FILE.open("w", encoding="utf-8") as f:
            json.dump(projects, f, indent=2)
    except OSError as e:
        st.error(f"Could not save project data: {e}")


def show_parse_error(raw):
    """Friendly error instead of dumping raw model output on screen."""
    truncated = raw is not None and not raw.rstrip().endswith(("}", "]", "```"))
    if truncated:
        st.error(
            "The AI response was cut off before it finished. This usually means the "
            "model was briefly overloaded. Click the button again to retry."
        )
    else:
        st.error("The AI returned an unexpected format. Please try again.")
    if raw:
        with st.expander("Technical details (for debugging)"):
            st.code(raw[:2000])


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "projects" not in st.session_state:
    st.session_state.projects = load_projects()
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = st.session_state.projects[0]["id"]
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None

projects = st.session_state.projects
project = next(
    (item for item in projects if item["id"] == st.session_state.current_project_id),
    projects[0],
)
st.session_state.current_project_id = project["id"]
project.setdefault("research_thread", {})
for _k in ["curiosity", "observation", "initial_question", "question_reasoning", "gap_reasoning", "methodology_reasoning"]:
    project["research_thread"].setdefault(_k, "")
project.setdefault("topic", {}).setdefault("defence", {})
project.setdefault("literature", {}).setdefault("defence", {})
project.setdefault("methodology", {}).setdefault("defence", {})
for _paper in project.get("literature", {}).get("papers", []): _paper.setdefault("student_notes", {})


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
def get_client():
    api_key = (st.session_state.get("api_key") or "").strip()
    if not api_key:
        try:
            api_key = (st.secrets.get("GOOGLE_API_KEY", "") or "").strip()
        except Exception:
            pass
    if not api_key:
        api_key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def get_model():
    return st.session_state.get("model", "gemini-3.5-flash-lite")


def ask_ai(user_prompt, max_tokens=4096, retries=3):
    """Generate content with automatic retry on temporary overloads (503/429)."""
    client = get_client()
    if client is None:
        st.warning("Add a Google AI API key in the sidebar to enable AI features.")
        return None

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=get_model(),
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=BASE_SYSTEM_PROMPT,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text or ""
        except Exception as e:
            last_error = e
            msg = str(e)
            # Retry only on temporary server-side issues
            if "503" in msg or "UNAVAILABLE" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt < retries:
                    wait = 5 * attempt  # 5s, 10s, 15s
                    st.toast(
                        f"Model busy (503). Retrying in {wait}s... "
                        f"(attempt {attempt}/{retries - 1})",
                        icon=":hourglass:",
                    )
                    import time
                    time.sleep(wait)
                    continue
            break

    st.error("The AI request failed after retries. Try again in a minute, or switch models in the sidebar.")
    with st.expander("Technical details"):
        st.code(str(last_error)[:1000])
    return None


def parse_json_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    return json.loads(cleaned)


def extract_pdf_text(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        parts = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(parts).strip()
    except Exception:
        return ""


def ai_available():
    return get_client() is not None


def ai_notice():
    if not ai_available():
        st.info("AI features are off until you add a Google AI API key in the sidebar. Forms still work.")


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
def page_dashboard():
    st.markdown(f'<div class="eyebrow">Research workspace · {RESEARCH_FIELD}</div>', unsafe_allow_html=True)
    scale_svg = (
        '<svg class="hero-watermark" viewBox="0 0 100 100" fill="none" '
        'stroke="#f3dfa8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="50" y1="10" x2="50" y2="80"/>'
        '<line x1="20" y1="22" x2="80" y2="22"/>'
        '<line x1="50" y1="80" x2="32" y2="92"/>'
        '<line x1="50" y1="80" x2="68" y2="92"/>'
        '<line x1="20" y1="22" x2="8" y2="45"/>'
        '<path d="M8 45 a12 10 0 0 0 24 0 Z"/>'
        '<line x1="80" y1="22" x2="68" y2="45"/>'
        '<path d="M68 45 a12 10 0 0 0 24 0 Z"/>'
        '<circle cx="50" cy="14" r="4"/>'
        '</svg>'
    )
    st.markdown(
        f'<div class="hero">'
        f'{scale_svg}'
        f'<div class="hero-badge">{RESEARCH_FIELD} research workspace</div>'
        f'<h1>{project["title"] or "Your next strong argument starts here."}</h1>'
        f'<p>Turn a legal idea into a focused question, evidence map, and defensible research design.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    stages = [
        ("Module 1 — Topic & Scope", "Shape your idea into a precise legal research question."),
        ("Module 2 — Literature Review", "Compare papers, authorities, arguments, and stated limitations."),
        ("Module 3 — Methodology", "Choose a defensible doctrinal, empirical, or comparative approach."),
    ]
    statuses = {
        "Module 1 — Topic & Scope": project["topic"]["status"],
        "Module 2 — Literature Review": project["literature"]["status"],
        "Module 3 — Methodology": project["methodology"]["status"],
    }
    completed = sum(1 for s in statuses.values() if s == "completed")
    overall = sum(0.33 for s in statuses.values() if s == "completed")
    overall += 0.17 if project["topic"]["status"] == "in_progress" else 0

    # Progress overview
    metric_items = [
        ("Stages done", f"{completed} / 3"),
        ("Papers uploaded", str(len(project["literature"]["papers"]))),
        ("Field", project.get("research_field", RESEARCH_FIELD)),
        ("AI mentor", "Connected" if ai_available() else "Key needed"),
    ]
    metric_html = '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1rem;">'
    for label, value in metric_items:
        metric_html += (
            '<div style="flex:1;min-width:140px;background:white;border:1px solid #dbe0ea;'
            'border-top:3px solid #c9a24a;border-radius:12px;padding:.7rem .9rem;">'
            f'<div style="font-size:.78rem;color:#5a6478;white-space:nowrap;">{label}</div>'
            f'<div style="font-size:1.3rem;font-weight:600;color:#0f1f3d;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{value}</div>'
            '</div>'
        )
    metric_html += '</div>'
    st.markdown(metric_html, unsafe_allow_html=True)

    st.progress(min(overall, 1.0), text=f"Overall progress — {STATUS_LABELS.get(statuses['Module 1 — Topic & Scope'])}")

    # Per-stage cards
    st.subheader("Your research path")
    cols = st.columns(3)
    for col, (name, description) in zip(cols, stages):
        status = statuses[name]
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(description)
                if status == "completed":
                    badge_bg, badge_fg, badge_text = "#e7f0e3", "#3d5c2f", STATUS_LABELS[status]
                elif status == "locked":
                    badge_bg, badge_fg, badge_text = "#eceef2", "#5a6478", STATUS_LABELS[status]
                else:
                    badge_bg, badge_fg, badge_text = "#fdf3dd", "#7a5c1e", STATUS_LABELS[status]
                st.markdown(
                    f'<span style="display:inline-block;background:{badge_bg};color:{badge_fg};'
                    f'border-radius:8px;padding:.3rem .7rem;font-size:.85rem;font-weight:500;">'
                    f'{badge_text}</span>',
                    unsafe_allow_html=True,
                )

    st.divider()
    st.subheader("🧭 Your Research Journey")
    rt = project.get("research_thread", {})
    q = project.get("topic", {}).get("selected_question") or rt.get("initial_question") or "Not defined yet"
    gap = project.get("literature", {}).get("gap_statement", {})
    gap_text = gap.get("gap") or gap.get("statement") or gap.get("summary") or "Still exploring the evidence"
    method = project.get("methodology", {}).get("paradigm") or "Not chosen yet"
    journey = [
        ("🧠 1. CURIOSITY", "What interests you?", rt.get("curiosity") or rt.get("observation") or "Start by recording what made you curious."),
        ("🔎 2. EVIDENCE", "What does existing research say?", f"{len(project.get('literature', {}).get('papers', []))} paper(s) collected. Analyse patterns before accepting a gap."),
        ("💡 3. RESEARCH GAP", "What is still unclear?", gap_text),
        ("🧪 4. METHODOLOGY", "How will you investigate it?", method),
    ]
    for title, question, detail in journey:
        with st.container(border=True):
            st.markdown(f"### {title}")
            st.caption(question)
            st.write(detail)
    with st.expander("🔗 See how your modules connect"):
        st.markdown("""
**Module 1 → Curiosity and Question**: You begin with something you noticed and refine it into a research direction.\n\n**Module 2 → Evidence and Gap**: You investigate what other researchers found and identify what is still uncertain.\n\n**Module 3 → Research Design**: You choose a method that can answer *your specific question* and respond to the evidence you discovered.\n\nThe AI mentor should use this journey to challenge your decisions rather than treating each module as a separate form.
""")
    st.caption(f"Current research question: {q}")

    st.divider()
    st.markdown(
        '<div class="notice"><strong>Academic integrity notice:</strong> this app helps you '
        "brainstorm, outline, and get feedback. You are responsible for verifying every source "
        "and complying with your institution's policies. The AI never fabricates citations, "
        "cases, or findings.</div>",
        unsafe_allow_html=True,
    )



# ---------------------------------------------------------------------------
# Supervisor validation and guidance
# ---------------------------------------------------------------------------
def supervisor_validate(step, response, context=""):
    """AI gate: validates relevance and explains what the student should do next."""
    if not response or not response.strip():
        return {"valid": False, "reason": "Your response is empty.", "explanation": "Write your own answer before continuing.", "example": "Answer the question directly using your research topic.", "next_action": "Try again."}
    prompt=f"""You are a careful research supervisor. Validate a student's response for this step: {step}.
Context: {context}
Student response: {response}
Decide whether it is relevant, meaningful and sufficient to move forward. Reject greetings, random text, copied unrelated content, one-word answers and answers that do not address the question. Do not reject imperfect academic writing if the student is genuinely attempting.
Return ONLY JSON: {{"valid": true/false, "reason":"...", "explanation":"specific explanation of what is missing or wrong", "example":"short example relevant to the task but not a full answer for their topic", "next_action":"clear instruction"}}"""
    raw=ask_ai(prompt, 1024)
    if not raw: return {"valid": True, "reason":"Saved for supervisor review.", "explanation":"AI validation is temporarily unavailable.", "example":"", "next_action":"You may continue, but review your answer carefully."}
    try: return parse_json_response(raw)
    except Exception: return {"valid": True, "reason":"Saved.", "explanation":"AI format issue; continue and review carefully.", "example":"", "next_action":"Continue."}

def show_guidance(title, explanation, example):
    with st.expander(f"💡 Supervisor guidance: {title}", expanded=False):
        st.write(explanation)
        if example: st.info("Example: " + example)

def gate_result(result):
    if result.get("valid"):
        st.success("Supervisor check passed: " + result.get("reason", "Your response addresses this step."))
        return True
    st.error("Please try again: " + result.get("reason", "Your response does not address the task."))
    st.info(result.get("explanation", "Answer the question directly."))
    if result.get("example"): st.caption("Example: " + result["example"])
    st.warning("Next step: " + result.get("next_action", "Revise your response and submit it again."))
    return False

# ---------------------------------------------------------------------------
# Page: Module 1 — Topic & Scope
# ---------------------------------------------------------------------------
def page_topic():
    st.title("Module 1 — Research Discovery")
    st.caption("Your supervisor guides you step by step. You cannot skip the thinking stages, but you will always get an explanation and example.")
    ai_notice(); rt=project["research_thread"]
    st.subheader("Step 1 — Start with curiosity")
    show_guidance("What interests you?", "Describe the research area that genuinely interests you and what part of it makes you curious.", "I am interested in how university students use generative AI because I have noticed it changing how assignments are completed.")
    rt["curiosity"]=st.text_area("What interests you about this topic?",rt.get("curiosity",""))
    rt["observation"]=st.text_area("What have you personally observed?",rt.get("observation",""),placeholder="Describe a real situation, pattern or problem you noticed.")
    if st.button("Check my curiosity and observation", disabled=not ai_available()):
        r=supervisor_validate("curiosity and observation", rt["curiosity"]+"\n"+rt["observation"]); rt["curiosity_check"]=r; save_projects(projects)
    if rt.get("curiosity_check"): gate_result(rt["curiosity_check"])

    if not rt.get("curiosity_check",{}).get("valid"): return
    st.subheader("Step 2 — Ask your own rough question")
    show_guidance("A rough research question", "Write one question you would genuinely like to investigate. It can be broad or imperfect; the supervisor will help you improve it.", "How does the use of generative AI influence independent learning among university students?")
    rt["initial_question"]=st.text_area("Write your first rough research question",rt.get("initial_question",""))
    rt["question_reasoning"]=st.text_area("Why do you think this question matters?",rt.get("question_reasoning",""))
    if st.button("Check my question", disabled=not ai_available()):
        r=supervisor_validate("rough research question and its importance", rt["initial_question"]+"\nReason: "+rt["question_reasoning"],rt["curiosity"]); rt["question_check"]=r; save_projects(projects)
    if rt.get("question_check"): gate_result(rt["question_check"])
    if not rt.get("question_check",{}).get("valid"): return

    st.subheader("Step 3 — Supervisor review")
    if st.button("🧠 Review My Thinking", disabled=not ai_available()):
        raw=ask_ai(f'''Act as a demanding but supportive research supervisor. Student curiosity: {rt["curiosity"]}\nObservation: {rt["observation"]}\nRough question: {rt["initial_question"]}\nReasoning: {rt["question_reasoning"]}\nDo not replace their work. Return ONLY JSON with strengths (list), challenges (list), supervisor_question, hint, research_mission.''')
        if raw:
            try: rt["review_feedback"]=parse_json_response(raw); project["topic"]["status"]="in_progress"; save_projects(projects)
            except: show_parse_error(raw)
    fb=rt.get("review_feedback",{})
    if not fb: return
    for x in fb.get("strengths",[]): st.success(x)
    for x in fb.get("challenges",[]): st.warning(x)
    st.info("Supervisor question: "+fb.get("supervisor_question","")); st.caption("Hint: "+fb.get("hint","")); st.markdown("### 🔎 Your research mission"); st.write(fb.get("research_mission","Find one relevant academic source and explain what you learned."))
    rt["mission_response"]=st.text_area("Bring back what you learned from your research mission",rt.get("mission_response",""),placeholder="Name the source, explain one finding, and say how it changed or supported your thinking.")
    if st.button("Check my research mission", disabled=not ai_available()):
        r=supervisor_validate("research mission report",rt["mission_response"],"Question: "+rt["initial_question"]); rt["mission_check"]=r; save_projects(projects)
    if rt.get("mission_check"): gate_result(rt["mission_check"])
    if not rt.get("mission_check",{}).get("valid"): return

    st.subheader("Step 4 — Compare and refine directions")
    project["broad_topic"]=st.text_area("Describe your general research area in your own words",project.get("broad_topic",rt["curiosity"]))
    if st.button("💡 Show possible directions", disabled=not ai_available()):
        raw=ask_ai(f'''Act as a supervisor. Area: {project["broad_topic"]}\nStudent question: {rt["initial_question"]}\nMission learning: {rt["mission_response"]}\nSuggest 3 focused directions for comparison only. Return ONLY JSON {{"questions":[{{"question":"...","rationale":"...","scope_notes":"...","feasibility_notes":"..."}}]}}''')
        if raw:
            try: project["topic"]["research_questions"]=parse_json_response(raw).get("questions",[]); save_projects(projects)
            except: show_parse_error(raw)
    qs=project["topic"].get("research_questions",[])
    if not qs:return
    for q in qs:
        with st.expander(q.get("question","Direction")): st.write(q.get("rationale","")); st.caption(q.get("scope_notes","")); st.caption(q.get("feasibility_notes",""))
    opts=[q.get("question","") for q in qs]; selected=st.selectbox("Choose the direction you want to defend and refine",opts)
    d=project["topic"].setdefault("defence",{})
    show_guidance("Defending a question", "Explain your reasoning. A good defence connects the question to a real problem, available evidence and a realistic study.", "It matters because universities need evidence about whether AI use supports or weakens independent learning.")
    d["importance"]=st.text_area("Why does this question matter?",d.get("importance","")); d["beneficiaries"]=st.text_area("Who could benefit?",d.get("beneficiaries","")); d["researchable"]=st.text_area("Why is it realistically researchable?",d.get("researchable","")); d["verification"]=st.text_area("What evidence would test your assumptions?",d.get("verification",""))
    if st.button("Supervisor check: can I finalize this direction?",disabled=not ai_available()):
        r=supervisor_validate("research question defence",json.dumps(d),"Question: "+selected); project["topic"]["defence_check"]=r; save_projects(projects)
    if project["topic"].get("defence_check"): gate_result(project["topic"]["defence_check"])
    if not project["topic"].get("defence_check",{}).get("valid"):return
    if st.button("Prepare a supervised proposal draft",disabled=not ai_available()):
        project["topic"]["selected_question"]=selected
        raw=ask_ai(f'''Create a structured proposal based on this student's completed work, without inventing sources. Question: {selected}\nCuriosity: {rt["curiosity"]}\nMission: {rt["mission_response"]}\nDefence: {json.dumps(d)}\nReturn ONLY JSON with working_title,research_problem,main_question,subquestions,objectives,scope_and_limitations,significance,suggested_approach,outline.''')
        if raw:
            try: project["topic"]["proposal"]=parse_json_response(raw); save_projects(projects)
            except: show_parse_error(raw)
    p=project["topic"].get("proposal",{})
    if p:
        st.subheader("Supervisor-assisted proposal — review and edit")
        for k,label in [("working_title","Working title"),("research_problem","Research problem"),("main_question","Main research question"),("scope_and_limitations","Scope & limitations"),("significance","Significance"),("suggested_approach","Suggested approach")]: p[k]=st.text_area(label,str(p.get(k,"")))
        if st.button("Save & complete Module 1",type="primary"):
            project["topic"]["proposal"]=p; project["topic"]["status"]="completed"; project["literature"]["status"]="available"; save_projects(projects); st.success("Module 1 complete. Your supervisor will now carry your question into Module 2.")


# ---------------------------------------------------------------------------
# Page: Module 2 — Literature Review
# ---------------------------------------------------------------------------
def page_literature():
    st.title("Module 2 — Literature Investigation")
    if project["literature"]["status"]=="locked": st.warning("Complete Module 1 first."); return
    ai_notice(); rt=project["research_thread"]; q=project["topic"].get("selected_question",rt.get("initial_question",""))
    st.info(f"Supervisor continuity — Your question: **{q}**\n\nYour job now is to test your thinking against evidence, not to prove your original idea.")
    st.subheader("Step 1 — Add evidence")
    files=st.file_uploader("Upload academic PDF papers",type="pdf",accept_multiple_files=True)
    if files:
        for uf in files:
            if any(x.get("filename")==Path(uf.name).name for x in project["literature"]["papers"]):continue
            data=uf.getvalue(); path=UPLOAD_DIR/f"{uuid.uuid4().hex}_{Path(uf.name).name}"; path.write_bytes(data)
            project["literature"]["papers"].append({"filename":Path(uf.name).name,"path":str(path),"entry":{},"student_notes":{}})
        save_projects(projects); st.success("Papers added. Read and give your first impression before asking AI to analyse them.")
    for i,paper in enumerate(project["literature"]["papers"]):
        st.markdown(f"### 📄 {paper['filename']}"); notes=paper.setdefault("student_notes",{})
        show_guidance("First reading", "Do not try to be perfect. Explain what you believe the study did, found and left unanswered.", "The study appears to survey students. I think its main finding is... I am unsure whether it explains long-term effects.")
        notes["about"]=st.text_area("What do YOU think this paper studies?",notes.get("about",""),key=f"about{i}")
        notes["finding"]=st.text_area("What do YOU think it found?",notes.get("finding",""),key=f"finding{i}")
        notes["question"]=st.text_area("What question or limitation did YOU notice?",notes.get("question",""),key=f"question{i}")
        if st.button("Check my reading",key=f"checkread{i}",disabled=not ai_available()): notes["check"]=supervisor_validate("initial reading of an academic paper",notes["about"]+"\n"+notes["finding"]+"\n"+notes["question"],q); save_projects(projects)
        if notes.get("check"): gate_result(notes["check"])
        if notes.get("check",{}).get("valid") and st.button("Ask supervisor to compare my reading with the paper",key=f"analyse{i}",disabled=not ai_available()):
            try: text="\n".join(page.extract_text() or "" for page in PdfReader(paper["path"]).pages)[:30000]
            except Exception: text=""
            raw=ask_ai(f'''Compare the student's reading with the uploaded paper text. Never invent. Student: {json.dumps(notes)}\nPaper text: {text}\nReturn ONLY JSON with author,year,methodology,findings,limitations,comparison_feedback,supervisor_question.''',8192)
            if raw:
                try: paper["entry"]=parse_json_response(raw); save_projects(projects)
                except: show_parse_error(raw)
        if paper.get("entry"):
            st.success("Supervisor comparison: "+paper["entry"].get("comparison_feedback","")); st.info("Question: "+paper["entry"].get("supervisor_question",""))
    analyzed=[p for p in project["literature"]["papers"] if p.get("entry")]
    if len(analyzed)<2:
        st.warning("Supervisor checkpoint: analyse at least 2 relevant papers before attempting to identify a pattern or gap."); return
    st.subheader("Step 2 — Discuss patterns, don't ask AI to invent a gap")
    rt["gap_reasoning"]=st.text_area("Which pattern, contradiction or limitation do YOU notice across the papers, and why does it matter?",rt.get("gap_reasoning",""))
    if st.button("Check my pattern",disabled=not ai_available()): rt["pattern_check"]=supervisor_validate("cross-paper pattern or limitation",rt["gap_reasoning"],q); save_projects(projects)
    if rt.get("pattern_check"): gate_result(rt["pattern_check"])
    if not rt.get("pattern_check",{}).get("valid"):return
    combined="\n\n".join(f"{p['filename']}: methodology={p['entry'].get('methodology')} findings={p['entry'].get('findings')} limitations={p['entry'].get('limitations')} student={p['student_notes'].get('question')}" for p in analyzed)
    if st.button("🔍 Discuss evidence with my supervisor",disabled=not ai_available()):
        raw=ask_ai(f'''Research question: {q}\nStudent's pattern: {rt["gap_reasoning"]}\nEvidence: {combined}\nAct as supervisor. Challenge weak inferences. Return ONLY JSON with patterns (list), contradictions (list), supervisor_challenge, possible_gap, evidence_notes, research_mission.''',4096)
        if raw:
            try: project["literature"]["supervisor_discussion"]=parse_json_response(raw); save_projects(projects)
            except: show_parse_error(raw)
    disc=project["literature"].get("supervisor_discussion",{})
    if disc:
        st.write("**Patterns:**",disc.get("patterns",[])); st.write("**Contradictions:**",disc.get("contradictions",[])); st.warning(disc.get("supervisor_challenge","")); st.info("Possible gap (verify): "+disc.get("possible_gap","")); st.caption("Research mission: "+disc.get("research_mission",""))
        d=project["literature"].setdefault("defence",{})
        d["known"]=st.text_area("What is already known? Cite the papers in your own words.",d.get("known","")); d["unclear"]=st.text_area("What remains unclear?",d.get("unclear","")); d["why_gap"]=st.text_area("Why is this a meaningful gap rather than simply 'not studied'?",d.get("why_gap","")); d["could_be_wrong"]=st.text_area("How could your assumption about this gap be wrong?",d.get("could_be_wrong",""))
        if st.button("Supervisor checkpoint: defend my gap",disabled=not ai_available()): project["literature"]["gap_check"]=supervisor_validate("evidence-based research gap defence",json.dumps(d),combined); save_projects(projects)
        if project["literature"].get("gap_check"): gate_result(project["literature"]["gap_check"])
        if project["literature"].get("gap_check",{}).get("valid"):
            project["literature"]["gap_statement"]={"existing_summary":d["known"],"known":d["known"],"unclear":d["unclear"],"proposed_gap":d["why_gap"],"significance":"Student defended this gap against the reviewed evidence."}
            if st.button("Save & complete Module 2",type="primary"):
                project["literature"]["status"]="completed"; project["methodology"]["status"]="available"; save_projects(projects); st.success("Module 2 complete. Your evidence and gap now guide Module 3.")


# ---------------------------------------------------------------------------
# Page: Module 3 — Methodology
# ---------------------------------------------------------------------------
def page_methodology():
    st.title("Module 3 — Research Design Lab")
    if project["methodology"]["status"]=="locked": st.warning("Complete Module 2 first."); return
    ai_notice(); rt=project["research_thread"]; q=project["topic"].get("selected_question",""); gap=project["literature"].get("gap_statement",{})
    st.info(f"Supervisor continuity — Question: **{q}**\n\nEvidence-based gap: **{gap.get('proposed_gap','')}**\n\nYour method must answer the question and respond to this evidence.")
    st.subheader("Step 1 — Decide what evidence you actually need")
    need=st.multiselect("What kind of information do you need?",["People's experiences or views","Numerical patterns or measurements","Laws, cases or documents","Comparison between systems or jurisdictions"])
    rt["methodology_reasoning"]=st.text_area("Why will this kind of evidence help answer your question?",rt.get("methodology_reasoning",""))
    show_guidance("Choosing evidence", "Start from the question, not from a fashionable method.", "If my question asks how people experience something, interviews may help explain experiences, while a survey may show broader patterns.")
    if st.button("Check my evidence reasoning",disabled=not ai_available()): rt["method_check"]=supervisor_validate("needed evidence and methodology reasoning",json.dumps({"need":need,"reason":rt["methodology_reasoning"]}),q); save_projects(projects)
    if rt.get("method_check"): gate_result(rt["method_check"])
    if not rt.get("method_check",{}).get("valid"):return
    paradigms=["Doctrinal Legal Research","Socio-Legal / Empirical Research","Comparative Research","Mixed / Combined Approach"]
    paradigm=st.radio("Choose the approach you currently think fits",paradigms,index=paradigms.index(project["methodology"].get("paradigm",paradigms[0])) if project["methodology"].get("paradigm") in paradigms else 0)
    st.subheader("Step 2 — Build and defend your design")
    md=project["methodology"].setdefault("defence",{})
    md["population"]=st.text_area("Who or what will you study, and why?",md.get("population","")); md["method"]=st.text_area("How will you collect/analyse evidence?",md.get("method","")); md["access"]=st.text_area("How will you realistically access participants or sources?",md.get("access","")); md["limitations"]=st.text_area("What can this method NOT tell you?",md.get("limitations","")); md["ethics"]=st.text_area("What ethical issues or permissions might arise?",md.get("ethics","")); md["time_skills"]=st.text_area("Why is this feasible with your time and skills?",md.get("time_skills",""))
    if st.button("🧠 Challenge my complete design",disabled=not ai_available()):
        raw=ask_ai(f'''Act as a demanding supervisor. Research question: {q}\nLiterature gap: {gap.get("proposed_gap","")}\nStudent evidence need: {need}\nApproach: {paradigm}\nDesign defence: {json.dumps(md)}\nCheck relevance, contradictions with Modules 1-2, feasibility and ethics. Return ONLY JSON with valid (boolean), strengths (list), problems (list), challenge_questions (list), revision_instruction, example_direction.''',4096)
        if raw:
            try: project["methodology"]["supervisor_review"]=parse_json_response(raw); save_projects(projects)
            except: show_parse_error(raw)
    review=project["methodology"].get("supervisor_review",{})
    if review:
        for x in review.get("strengths",[]):st.success(x)
        for x in review.get("problems",[]):st.error(x)
        for x in review.get("challenge_questions",[]):st.warning("Supervisor asks: "+x)
        st.info(review.get("revision_instruction","")); st.caption("Example direction: "+review.get("example_direction",""))
    if review.get("valid"):
        if st.button("Supervisor check: finalize methodology",disabled=not ai_available()): project["methodology"]["final_check"]=supervisor_validate("complete and feasible research methodology",json.dumps(md),q+" Gap: "+gap.get("proposed_gap","")); save_projects(projects)
        if project["methodology"].get("final_check"): gate_result(project["methodology"]["final_check"])
        if project["methodology"].get("final_check",{}).get("valid") and st.button("Save & complete Module 3",type="primary"):
            project["methodology"]["paradigm"]=paradigm; project["methodology"]["plan"]={"paradigm":paradigm,"evidence_needed":need,"student_design":md}; project["methodology"]["status"]="completed"; save_projects(projects); st.success("All modules complete. You now have a documented research journey, evidence trail and defended methodology.")


# ---------------------------------------------------------------------------
# Sidebar & navigation
# ---------------------------------------------------------------------------
PAGES = {
    "Dashboard": page_dashboard,
    "Module 1 — Research Discovery": page_topic,
    "Module 2 — Literature Investigation": page_literature,
    "Module 3 — Research Design Lab": page_methodology,
}

with st.sidebar:
    st.markdown(f"**{APP_NAME}**")
    st.caption("Local research workspace")

    project_labels = [item.get("title") or "Untitled research project" for item in projects]
    current_index = next(
        (index for index, item in enumerate(projects) if item["id"] == project["id"]), 0
    )
    selected_index = st.selectbox(
        "Open a project", range(len(projects)),
        index=current_index, format_func=lambda i: project_labels[i],
        label_visibility="collapsed",
    )
    if projects[selected_index]["id"] != project["id"]:
        st.session_state.current_project_id = projects[selected_index]["id"]
        st.rerun()

    if st.button("New research project", use_container_width=True):
        new_project = default_project("New legal research project")
        projects.append(new_project)
        st.session_state.current_project_id = new_project["id"]
        save_projects(projects)
        st.rerun()

    st.divider()
    choice = st.radio("Navigate", list(PAGES.keys()))

    with st.expander("AI settings"):
        model_name = st.selectbox("Free Gemini model", list(MODEL_OPTIONS.keys()))
        st.session_state.model = MODEL_OPTIONS[model_name]
        st.session_state.api_key = st.text_input(
            "Google AI API key",
            value=st.session_state.get("api_key", ""),
            type="password",
            help="Used only for this browser session; never saved to your project file.",
        )
        st.caption("Key stays in this browser session only.")
        if st.session_state.get("api_key"):
            st.success("API key active.")
        else:
            st.caption("No key set — AI features are off.")

PAGES[choice]()