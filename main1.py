import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Retriever from your vector DB
from vector1 import retriever

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="F1 2025 Assistant", page_icon="🏎️", layout="centered")
st.title("🏎️ Formula 1 2025 Chatbot")
st.write("Ask anything about the 2025 F1 race results.")

# =========================
# LLM CHAIN
# =========================
@st.cache_resource
def get_chain():
    model = OllamaLLM(model="gemma3:latest")

    template = """
    You are a factual assistant answering questions about the 2025 Formula 1 season.

    Rules:
    - Use ONLY the provided race result records
    - Do NOT invent drivers or results
    - Do NOT calculate statistics unless clearly stated
    - If the answer is missing, say:
      "The dataset does not contain this information."
    - Keep answers short and factual

    Dataset context:
    - Track = Grand Prix name
    - Position = finishing position
    - Driver = driver name
    - Team = constructor
    - Points = points scored
    - Fastest Lap Time may be included

    F1 records:
    {records}

    Question:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

chain = get_chain()

# =========================
# CHAT MEMORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# USER INPUT
# =========================
if question := st.chat_input("Example: Who won the Australia GP 2025?"):

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Analyzing race results..."):

            # Retrieve documents
            docs = retriever.invoke(question)

            # 🔴 IMPORTANT FIX: Convert docs → text
            records = "\n".join([d.page_content for d in docs])

            response = chain.invoke({
                "records": records,
                "question": question
            })

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
