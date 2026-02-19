import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector1 import retriever

# ================= PAGE CONFIG =================
st.set_page_config(page_title="F1 AI Assistant", layout="wide")

# ================= GLOBAL STYLES =================
st.markdown("""
<style>

/* ===== GLOBAL DARK THEME ===== */
html, body, [class*="css"] {
    background: linear-gradient(180deg, #0a0a0a 0%, #111111 100%);
    color: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== TITLE ===== */
.f1-title {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-align: center;
    margin-bottom: 0;
}

.f1-sub {
    text-align: center;
    color: #aaaaaa;
    margin-bottom: 25px;
}

/* ===== START LIGHTS ===== */
.lights {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin: 15px 0 25px 0;
}

.light {
    width: 18px;
    height: 18px;
    background: #300;
    border-radius: 50%;
    box-shadow: 0 0 10px #000 inset;
    animation: startLights 3s infinite;
}

.light:nth-child(1){animation-delay:0s}
.light:nth-child(2){animation-delay:.3s}
.light:nth-child(3){animation-delay:.6s}
.light:nth-child(4){animation-delay:.9s}
.light:nth-child(5){animation-delay:1.2s}

@keyframes startLights {
    0% { background:#300; box-shadow:none;}
    50% { background:#ff1e1e; box-shadow:0 0 12px #ff1e1e;}
    100% { background:#300; }
}

/* ===== TRACK ===== */
.track {
    position: relative;
    height: 70px;
    margin-bottom: 25px;
    overflow: hidden;
}

.track-line {
    position: absolute;
    bottom: 12px;
    width: 100%;
    height: 3px;
    background: repeating-linear-gradient(
        to right,
        #fff 0 30px,
        transparent 30px 60px
    );
    opacity: 0.6;
}

/* ===== F1 CAR ANIMATION ===== */
.f1-car {
    position: absolute;
    bottom: 0;
    left: -120px;
    font-size: 2.5rem;
    animation: drive 5s linear infinite;
    transform: scaleX(-1); /* flips car forward */
}


@keyframes drive {
    0% { left: -120px; }
    100% { left: 110%; }
}

/* ===== GLASS CHAT ===== */
.chat-container {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ===== INPUT ===== */
.stChatInput input {
    background: #1a1a1a !important;
    color: white !important;
}

/* ===== ASSISTANT MESSAGES ===== */
[data-testid="stChatMessageContent"] {
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="f1-title">FORMULA 1 AI ANALYST</div>', unsafe_allow_html=True)
st.markdown('<div class="f1-sub">Real race data. No hallucinations.</div>', unsafe_allow_html=True)

# ===== START LIGHTS =====
st.markdown("""
<div class="lights">
  <div class="light"></div>
  <div class="light"></div>
  <div class="light"></div>
  <div class="light"></div>
  <div class="light"></div>
</div>
""", unsafe_allow_html=True)

# ===== TRACK + CAR =====
st.markdown("""
<div class="track">
  <div class="f1-car">🏎️</div>
  <div class="track-line"></div>
</div>
""", unsafe_allow_html=True)

# ================= LLM =================
@st.cache_resource
def get_chain():
    model = OllamaLLM(model="gemma3:latest")

    template = """
You are a professional Formula 1 data analyst.

Rules:
- Use ONLY the provided dataset records
- No guessing or hallucination
- If missing → say: "The dataset does not contain this information."
- Be concise and factual
- Prefer bullet points when useful

Records:
{records}

Question:
{question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

chain = get_chain()

# ================= CHAT STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== CHAT BOX =====
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= INPUT =================
if question := st.chat_input("Ask about races, seasons, circuits, winners..."):

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing telemetry..."):
            records = retriever.invoke(question)
            response = chain.invoke({"records": records, "question": question})
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown('</div>', unsafe_allow_html=True)

