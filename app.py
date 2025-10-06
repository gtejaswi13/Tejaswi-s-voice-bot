# app.py
import os
import re
import json
import glob
import traceback
from uuid import uuid4
from collections import deque
from typing import List, Tuple

import gradio as gr
from gtts import gTTS
import speech_recognition as sr

# LangChain bits
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========== Load JSON knowledge ==========
with open("knowledge.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)

def flatten_json_to_docs(obj, path="root"):
    docs = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            docs += flatten_json_to_docs(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            docs += flatten_json_to_docs(v, f"{path}[{i}]")
    else:
        content = str(obj)
        if content.strip():
            docs.append(Document(page_content=content, metadata={"path": path}))
    return docs

raw_docs = flatten_json_to_docs(knowledge)

HELLO_RX = re.compile(r"\b(hi|hello|hey|good\s*(morning|afternoon|evening))\b", re.I)
THANKS_RX = re.compile(r"\b(thanks|thank\s*you|tysm|thank\s*u)\b", re.I)

def _pleasant_closing() -> str:
    """Return a warm closing when user says thanks."""
    return "You’re welcome! I had a great time chatting"

# Smaller chunks = more atomic facts
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=60,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
)
docs = splitter.split_documents(raw_docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(docs, embeddings)

# ========== LLM ==========
groq_api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("VOICE_BOT")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.2, max_tokens=350, groq_api_key=groq_api_key)

# ========== Light helpers ==========
_recent_paths = deque(maxlen=8)     # to avoid reusing same JSON paths
_recent_answers = deque(maxlen=8)   # to avoid repeating phrasing

def _sanitize_response(text: str) -> str:
    # Remove boilerplate if any sneaks in; strip surrounding quotes/backticks
    lines = [ln for ln in (text or "").splitlines() if not re.match(r"^\s*(here'?s|in a simple, warm tone)", ln, re.I)]
    text = " ".join(ln.strip() for ln in lines if ln.strip())
    text = re.sub(r'^[`"“]+|[`"”]+$', "", text).strip()
    return text

def _token_set(s: str) -> set:
    return set(t.lower() for t in re.findall(r"[a-z0-9]+", s or ""))

def retrieve(question: str, k: int = 10) -> List[Tuple[str, str, float]]:
    """
    Return [(snippet_text, json_path, score)].
    We add a small diversity penalty to paths used very recently.
    """
    hits = vectordb.similarity_search_with_score(question, k=k*2)
    results = []
    seen = set()
    for doc, score in hits:
        txt = (doc.page_content or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        path = (doc.metadata or {}).get("path", "root")
        # small penalty if path was used recently (discourage repetition)
        if path in _recent_paths:
            score += 0.12
        results.append((txt, path, float(score)))
    # sort by adjusted score (lower is closer in FAISS)
    results.sort(key=lambda x: x[2])
    # basic MMR-ish diversity: keep top unique snippets (by word overlap)
    picked = []
    for t, p, s in results:
        if len(picked) >= k:
            break
        too_similar = any(
            len(_token_set(t) & _token_set(t2)) / max(len(_token_set(t)), 1) > 0.7
            for (t2, _, _) in picked
        )
        if not too_similar:
            picked.append((t, p, s))
    return picked[:k]

# ========== Grounded answering with verification ==========
SYSTEM_RULES = (
    "You are Gummadi Tejaswi speaking in first person (I/me). "
    "Answer ONLY using the provided context snippets. "
    "Never add facts that are not explicitly stated in the context. "
    "If the context does not contain the answer, reply exactly: I don't have that info yet.\n\n"
    "Style: warm, concise, natural one-liners unless a short paragraph is needed. "
    "Match the question intent (who/what/when/where/how many/yes-no). "
    "Avoid repeating the same sentence you used earlier if possible."
)

def build_context_block(snippets: List[Tuple[str, str, float]]) -> str:
    # Provide numbered context with JSON paths to help the LLM stay precise
    lines = []
    for i, (text, path, _) in enumerate(snippets, 1):
        lines.append(f"[{i}] ({path}) {text}")
    return "\n".join(lines)

def generate_grounded_answer(question: str) -> Tuple[str, List[str]]:
    """
    1) Retrieve context
    2) Ask LLM to answer strictly with that context
    3) Verify with a second LLM check that nothing is invented
    """
    snippets = retrieve(question, k=8)
    if not snippets:
        return "I don't have that info yet.", []

    context_block = build_context_block(snippets)
    used_paths = [p for (_, p, _) in snippets]

    # Provide previous answers to encourage varied phrasing (not rules; just guidance)
    prev = " | ".join(list(_recent_answers)[-3:]) if _recent_answers else ""

    draft = llm([
        SystemMessage(content=SYSTEM_RULES + (f"\n\nRecent phrasings to avoid repeating verbatim: {prev}" if prev else "")),
        HumanMessage(content=f"Question: {question}\n\nContext Snippets:\n{context_block}\n\nAnswer:")
    ]).content or ""

    draft = _sanitize_response(draft).strip() or "I don't have that info yet."

    # Quick form check: if the model tried to add preambles or quotes, sanitize already did.

    # Verification: ensure every claim is supported by the snippets
    verifier_prompt = (
        "You must act as a strict fact verifier.\n"
        "Given the context snippets and the draft answer, determine if EVERY factual claim in the draft "
        "is explicitly supported by the context text.\n"
        "Respond with exactly one word: YES or NO.\n"
    )
    verdict = llm([
        SystemMessage(content=verifier_prompt),
        HumanMessage(content=f"Context:\n{context_block}\n\nDraft:\n{draft}")
    ]).content.strip().upper()

    if verdict != "YES":
        # If unsupported, do one more constrained rewrite attempt:
        retry = llm([
            SystemMessage(content=SYSTEM_RULES + "\nRewrite ONLY with supported facts; if unsure, reply exactly: I don't have that info yet."),
            HumanMessage(content=f"Question: {question}\n\nContext:\n{context_block}\n\nAnswer:")
        ]).content or ""
        retry = _sanitize_response(retry).strip()
        if not retry or retry.lower().startswith("i don't") or retry.upper() == "I DON'T HAVE THAT INFO YET.":
            return "I don't have that info yet.", used_paths
        # Verify again
        verdict2 = llm([
            SystemMessage(content=verifier_prompt),
            HumanMessage(content=f"Context:\n{context_block}\n\nDraft:\n{retry}")
        ]).content.strip().upper()
        if verdict2 != "YES":
            return "I don't have that info yet.", used_paths
        draft = retry

    return draft, used_paths

# ========== Deterministic intro (still LLM-based, but fully grounded) ==========
INTRO_UTTERANCES = re.compile(r"\b(tell me about yourself|introduce yourself|about you|yourself)\b", re.I)

def llm_grounded_intro(question: str) -> Tuple[str, List[str]]:
    """
    Use the same retrieval + strict LLM rules to produce a short intro,
    but ask for 1–2 sentences max.
    """
    snippets = retrieve("intro about me biography summary", k=8)
    context_block = build_context_block(snippets)
    used_paths = [p for (_, p, _) in snippets]

    intro_rules = (
        SYSTEM_RULES +
        "\nFormat the intro in 1–2 short sentences. Do not add details not present. No lists."
    )
    resp = llm([
        SystemMessage(content=intro_rules),
        HumanMessage(content=f"User asked: {question}\n\nContext Snippets:\n{context_block}\n\nAnswer:")
    ]).content or ""
    resp = _sanitize_response(resp).strip() or "I don't have that info yet."
    return resp, used_paths

# ========== Voice + text handler ==========
def voice_bot(audio=None, text=None):
    try:
        # Input
        if audio:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio) as source:
                voice_data = recognizer.record(source)
                user_input = recognizer.recognize_google(voice_data)
        elif text and text.strip():
            user_input = text.strip()
        else:
            return "⚠️ Please ask a question by typing or speaking.", None

        q_lower = user_input.lower()

        # Self-intro requests still go through LLM but grounded
        if INTRO_UTTERANCES.search(q_lower):
            response, used_paths = llm_grounded_intro(user_input)
        else:
            response, used_paths = generate_grounded_answer(user_input)

        if not response:
            response = "I don't have that info yet."

        # Privacy guard: only reveal email/phone when explicitly asked
        wants_phone = any(k in q_lower for k in ["phone", "mobile", "contact number"])
        wants_email = any(k in q_lower for k in ["email", "mail id", "mail address", "email id"])
        phone = knowledge.get("personal_info", {}).get("phone")
        email = knowledge.get("personal_info", {}).get("email")

        if not wants_phone and phone:
            response = response.replace(str(phone), "[phone withheld]")
        if not wants_email and email:
            response = response.replace(str(email), "[email withheld]")

        # Track recent to reduce repetition next turns
        _recent_answers.append(response)
        for p in used_paths:
            _recent_paths.append(p)

        # TTS output
        audio_file = None
        try:
            for old in glob.glob("response_*.mp3"):
                try:
                    os.remove(old)
                except:
                    pass
            tts = gTTS(response)
            audio_file = f"response_{uuid4().hex}.mp3"
            tts.save(audio_file)
        except Exception as tts_err:
            print("gTTS error:", tts_err)
            traceback.print_exc()
            audio_file = None

        return response, audio_file

    except Exception as e:
        msg = str(e)
        print("Unhandled error:", msg)
        traceback.print_exc()
        return f"⚠️ Error: {msg}", None

# ========== Gradio UI ==========
with gr.Blocks() as demo:
    gr.Markdown("### 🎙️Hi, I'm Tejaswi.")

    with gr.Row():
        voice_in = gr.Audio(label="Ask your question", type="filepath")
        text_in = gr.Textbox(label="Or type your question")

    btn = gr.Button("Submit")
    text_out = gr.Textbox(label="Tejaswi Says")
    voice_out = gr.Audio(label="Voice Reply", type="filepath", autoplay=True)

    btn.click(fn=voice_bot, inputs=[voice_in, text_in], outputs=[text_out, voice_out])

if __name__ == "__main__":
    demo.queue().launch()
