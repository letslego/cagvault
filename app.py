import streamlit as st
from config import Config
from knowledge import KnowledgeType, load_from_file, load_from_url
from models import create_llm, BaseChatModel
from chatbot import ChunkType, ask, create_context_cache, clear_cache
from kvcache import get_kv_cache

st.set_page_config(
    page_title="CogVault CAG", layout="centered", initial_sidebar_state="expanded", page_icon="🌵"
)
 
st.header("🌵 CAG Agentic Chat")
st.subheader("Completely local and private chat with your documents")

@st.cache_resource(show_spinner=False)
def create_chatbot() -> BaseChatModel:
    return create_llm(Config.MODEL)
 
def delete_source(source_id):
    del st.session_state.sources[source_id]
    
if "sources" not in st.session_state:
    st.session_state.sources = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0
if "context_cache_id" not in st.session_state:
    st.session_state.context_cache_id = None
    
with st.sidebar:
    st.title("CAG Vault")
    
    # Show cache stats
    kv_cache = get_kv_cache()
    stats = kv_cache.get_stats()
    if stats["total_entries"] > 0:
        with st.expander("📊 Cache Stats", expanded=False):
            st.metric("Cached Contexts", stats["total_entries"])
            st.metric("Total Tokens", stats["total_tokens"])
            st.metric("Cache Hits", stats["total_hits"])
            if st.button("🧹 Clear Cache"):
                clear_cache()
                st.session_state.context_cache_id = None
                st.rerun()
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload a Document",
        type=Config.UI.ALLOWED_FILE_TYPES,
        accept_multiple_files=False,
        label_visibility="collapsed",
        key=f"pdf_uploader_{st.session_state.upload_counter}",
    )
    if uploaded_file is not None:
        st.session_state.upload_counter += 1
        st.info(f"CAG is learning from {uploaded_file.name}...")
        source = load_from_file(uploaded_file.name, uploaded_file.read())
        st.session_state.sources[source.id] = source
        # Invalidate cache when sources change
        st.session_state.context_cache_id = None
        st.rerun()
        
        st.markdown("---")
 
    with st.form(key="add_url_form", clear_on_submit=True):
        url_input = st.text_input(
            "Or, paste a Web Page Link (URL)",
            placeholder="e.g., https://agenticknowhow.com",
        )
        submit = st.form_submit_button("Add Web Page")
        if submit:
            if url_input and url_input.startswith(("http://", "https://")):
                st.info(f"CAG is learning from {url_input}...")
                source = load_from_url(url_input)
                st.session_state.sources[source.id] = source
                # Invalidate cache when sources change
                st.session_state.context_cache_id = None
                st.rerun()
            elif url_input:
                st.error("Please enter a valid URL (http:// or https://)")
            else:
                st.warning("Please paste a web page link first.")
                
    st.subheader("Your documents")
    if not st.session_state.sources:
        st.info("Add a document above to start")
    else:
        source_to_delete = None
        source_list_container = st.container()
        with source_list_container:
            for source_id, source in st.session_state.sources.items():
                display_name = source.name
                if source.type == KnowledgeType.URL and len(display_name) > 35:
                    display_name = display_name[:30] + "..."
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    col1.text(display_name)
                with col2:
                    button_key = f"delete_{source_id}"
                    if col2.button("🗑️", key=button_key, help="Remove this source"):
                        source_to_delete = source_id
        if source_to_delete:
            delete_source(source_to_delete)
            st.rerun()
if not st.session_state.sources:
    st.info("Add your documents in the sidebar to start chatting", icon="👈")
elif st.session_state.sources:
    # Create context cache if needed
    if st.session_state.context_cache_id is None:
        with st.spinner("🔄 Caching context for faster queries..."):
            st.session_state.context_cache_id = create_context_cache(st.session_state.sources)
    
    for message in st.session_state.messages:
        avatar_icon = "🤖" if message["role"] == "assistant" else "🐧"
        with st.chat_message(message["role"], avatar=avatar_icon):
            thinking_content = message.get("thinking")
            if thinking_content:
                with st.expander("CogVault's thoughts", expanded=False):
                    st.markdown(thinking_content)
 
            st.markdown(message["content"])
            
    chat_enabled = len(st.session_state.sources) > 0
 
    if prompt := st.chat_input(
        "Ask CogVault about your documents...",
        key="chat_input",
        disabled=not chat_enabled,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt, "thinking": None})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant", avatar="🤖"):
            think_expander_placeholder = st.empty()
            message_placeholder = st.empty()
 
            full_response_content = ""
            thinking_content_buffer = ""
 
            llm = create_chatbot()
            query = st.session_state.messages[-1]["content"]
            history = st.session_state.messages[:-1]
            for chunk_data in ask(query, history, st.session_state.sources, llm):
                chunk_type = chunk_data.type
                chunk_content = chunk_data.content
 
                if chunk_type == ChunkType.CONTENT:
                    full_response_content += chunk_content
                    message_placeholder.markdown(full_response_content + "▌")
                elif chunk_type == ChunkType.START_THINK:
                    with think_expander_placeholder.container():
                        with st.expander("CogVault thoughts...", expanded=True):
                            st.markdown("...")
                elif chunk_type == ChunkType.THINKING:
                    thinking_content_buffer += chunk_content
                    with think_expander_placeholder.container():
                        with st.expander("CogVault thoughts...", expanded=True):
                            st.markdown(thinking_content_buffer + "▌")
                elif chunk_type == ChunkType.END_THINK:
                    with think_expander_placeholder.container():
                        with st.expander("CogVault thoughts...", expanded=True):
                            st.markdown(thinking_content_buffer)
                else:
                    st.error("Unknown chunk type received from the model.")
 
            message_placeholder.markdown(full_response_content)
            think_expander_placeholder.empty()
 
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response_content,
                    "thinking": thinking_content_buffer if thinking_content_buffer else None,
                }
            )
            st.rerun()