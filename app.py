import streamlit as st
import asyncio
import os
from src.config import HybridConfig
from src.agent import AgentBrain
from src.hybrid_rag_system import HybridRAGSystem

st.set_page_config(page_title="Local RAG", layout="wide")

# --- Инициализация ---
@st.cache_resource
def get_engine():
    cfg = HybridConfig()
    rag = HybridRAGSystem(cfg)
    agent = AgentBrain(cfg, rag)
    return agent, rag, cfg

try:
    agent, rag, cfg = get_engine()
except Exception as e:
    st.error(f"Failed to init system: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("🎛️ Control")
    stats = rag.get_stats()
    st.metric("Documents in DB", stats["count"])

    uploaded_files = st.file_uploader("Upload Docs (.txt, .md)", accept_multiple_files=True)
    if uploaded_files and st.button("Index Files"):
        save_dir = cfg.data_dir / "uploads"
        save_dir.mkdir(parents=True, exist_ok=True)

        with st.status("Indexing...") as status:
            for f in uploaded_files:
                with open(save_dir / f.name, "wb") as w:
                    w.write(f.getbuffer())

            rag.index_documents(str(save_dir))
            status.update(label="Done!", state="complete")
        st.rerun()

# --- Chat ---
st.title("🤖 Local RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.text(s['text'][:200] + "...")

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # expanded=True заставляет окно быть открытым по умолчанию, чтобы видеть чанки сразу
        status = st.status("Thinking process...", expanded=True)

        def ui_cb(step, detail):
            # Используем markdown для красивого отображения цитат и жирного текста
            status.markdown(f"**{step}**")
            status.markdown(detail)

        try:
            response = asyncio.run(agent.run(prompt, callback=ui_cb))

            # Меняем статус на завершенный, но не закрываем автоматически, если хотите видеть логи
            status.update(label="Complete", state="complete", expanded=False)

            st.markdown(response["answer"])

            # (Опционально) Сохраняем источники, чтобы они были и в истории
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"]
            })
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Error: {e}")

