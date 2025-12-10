import streamlit as st
from config import Config
from knowledge import KnowledgeType, load_from_file, load_from_url
from models import create_llm, BaseChatModel
from chatbot import ChunkType, ask, create_context_cache, clear_cache
from kvcache import get_kv_cache

# Import Claude Skills PDF parser
from skills.pdf_parser.enhanced_parser import get_enhanced_parser
from skills.pdf_parser.ner_search import get_searchable_parser

st.set_page_config(
    page_title="CAG", layout="centered", initial_sidebar_state="expanded", page_icon="🌵"
)
 
st.header("🌵 CAG Agentic Chat")
st.subheader("Completely local and private chat with your documents")

@st.cache_resource(show_spinner=False)
def create_chatbot() -> BaseChatModel:
    return create_llm(Config.MODEL)

@st.cache_resource(show_spinner=False)
def init_pdf_parser():
    """Initialize enhanced PDF parser with NER and search."""
    return get_enhanced_parser()

@st.cache_resource(show_spinner=False)
def init_search_parser():
    """Initialize searchable parser."""
    return get_searchable_parser()
 
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
if "message_source_ids" not in st.session_state:
    st.session_state.message_source_ids = set()
if "parsed_documents" not in st.session_state:
    st.session_state.parsed_documents = {}  # Store parsed PDF metadata
    
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
        
        # Save file temporarily and parse with Claude Skills
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Load file into knowledge base
            source = load_from_file(uploaded_file.name, uploaded_file.read())
            st.session_state.sources[source.id] = source
            st.session_state.message_source_ids = {source.id}
            st.session_state.context_cache_id = None
            
            # Parse with Claude Skills for enhanced processing
            with st.spinner("🔄 Parsing PDF with Claude Skills..."):
                pdf_parser = init_pdf_parser()
                search_parser = init_search_parser()
                
                # Extract sections
                extraction_result = pdf_parser.parse_and_extract_sections(tmp_path)
                
                # Store parsed metadata
                st.session_state.parsed_documents[source.id] = {
                    'document_name': uploaded_file.name,
                    'document_id': extraction_result.get('document_id'),
                    'sections_count': extraction_result.get('sections_count', 0),
                    'extraction_result': extraction_result
                }
                
                # Show extraction summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Pages", extraction_result.get('pages_count', 0))
                with col2:
                    st.metric("📑 Sections", extraction_result.get('sections_count', 0))
                with col3:
                    st.metric("🔍 Entities", extraction_result.get('entities_found', 0))
                
            st.success(f"✅ {uploaded_file.name} parsed and ready for Q&A!")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        st.rerun()
        
        st.markdown("---")
 
    with st.form(key="add_url_form", clear_on_submit=True):
        url_input = st.text_input(
            "Or, paste a Web Page Link (URL)",
            placeholder="e.g., https://nammu21.com",
        )
        submit = st.form_submit_button("Add Web Page")
        if submit:
            if url_input and url_input.startswith(("http://", "https://")):
                st.info(f"CAG is learning from {url_input}...")
                source = load_from_url(url_input)
                st.session_state.sources[source.id] = source
                # Update messages to reference only the new documents
                st.session_state.message_source_ids = {source.id}
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
        # Show parsed document information
        if st.session_state.parsed_documents:
            st.markdown("**🔬 Parsed Document Analysis**")
            for source_id, parsed_info in st.session_state.parsed_documents.items():
                with st.expander(f"📋 {parsed_info['document_name']}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sections", parsed_info.get('sections_count', 0))
                    with col2:
                        entities = parsed_info.get('extraction_result', {}).get('entities_found', 0)
                        st.metric("Entities", entities)
                    with col3:
                        doc_id = parsed_info.get('extraction_result', {}).get('document_id', 'N/A')
                        st.caption(f"ID: {doc_id[:8]}...")
                    
                    # Sections, Search, and NER views
                    sections_tab, search_tab, entities_tab = st.tabs(["📄 Sections", "🔍 Search", "🏷️ Entities"])

                    with sections_tab:
                        parser = init_pdf_parser()
                        sections = parser.memory.get_document_sections(doc_id) if doc_id else []

                        def summarize(text: str, limit: int = 240) -> str:
                            """Compact summary using leading sentence or clipped text."""
                            cleaned = " ".join(part.strip() for part in text.splitlines() if part.strip())
                            if len(cleaned) <= limit:
                                return cleaned or "(no content)"
                            clipped = cleaned[:limit].rsplit(" ", 1)[0]
                            return f"{clipped}…"

                        coverage_message = parsed_info.get("extraction_result", {}).get("coverage_message")
                        if coverage_message:
                            st.caption(coverage_message)

                        if sections:
                            for section in sections:
                                meta = section.metadata
                                page_display = f"p. {meta.start_page}-{meta.end_page}" if (meta.start_page and meta.end_page) else f"p. {meta.page_range or meta.page_estimate}"
                                with st.expander(f"{meta.title} ({page_display})", expanded=False):
                                    st.markdown(
                                        f"**Level:** {meta.level} | **Pages:** {meta.start_page}–{meta.end_page} | "
                                        f"**Words:** {meta.word_count:,} | **Tables:** {bool(meta.has_tables)}"
                                    )
                                    st.markdown(summarize(section.content))
                        else:
                            st.info("Sections not available for this document yet.")
                    
                    with search_tab:
                        search_query = st.text_input(
                            "Search in document",
                            key=f"search_{source_id}",
                            placeholder="Find text..."
                        )
                        if search_query:
                            pdf_parser = init_pdf_parser()
                            results = pdf_parser.search_document(
                                doc_id, 
                                search_query
                            )
                            st.json(results, expanded=False)
                    
                    with entities_tab:
                        entity_type_filter = st.selectbox(
                            "Filter by entity type",
                            ["All", "MONEY", "DATE", "PARTY", "AGREEMENT", "PERCENTAGE"],
                            key=f"entity_filter_{source_id}"
                        )
                        search_parser = init_search_parser()
                        if entity_type_filter != "All":
                            entities = search_parser.search_engine.search_entities(doc_id, entity_type_filter)
                        else:
                            entities = search_parser.search_engine.entities_by_document.get(doc_id, [])
                        
                        if entities:
                            st.write(f"**Found {len(entities)} {entity_type_filter} entities**")
                            for entity in entities[:10]:
                                st.caption(f"**{entity.type}**: {entity.text}")
                        else:
                            st.info(f"No {entity_type_filter} entities found")
            
            st.markdown("---")
        
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
            st.session_state.message_source_ids.discard(source_to_delete)
            st.session_state.context_cache_id = None
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
                with st.expander("CAG's thoughts", expanded=False):
                    st.markdown(thinking_content)
 
            st.markdown(message["content"])
            
    chat_enabled = len(st.session_state.sources) > 0
 
    if prompt := st.chat_input(
        "Ask CAG about your documents...",
        key="chat_input",
        disabled=not chat_enabled,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt, "thinking": None})
        # Update message source IDs to current documents
        st.session_state.message_source_ids = set(st.session_state.sources.keys())
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
            # Filter sources to only include current message document references
            filtered_sources = {
                sid: source for sid, source in st.session_state.sources.items() 
                if sid in st.session_state.message_source_ids
            }
            for chunk_data in ask(query, history, filtered_sources, llm):
                chunk_type = chunk_data.type
                chunk_content = chunk_data.content
 
                if chunk_type == ChunkType.CONTENT:
                    full_response_content += chunk_content
                    message_placeholder.markdown(full_response_content + "▌")
                elif chunk_type == ChunkType.START_THINK:
                    with think_expander_placeholder.container():
                        with st.expander("CAG thoughts...", expanded=True):
                            st.markdown("...")
                elif chunk_type == ChunkType.THINKING:
                    thinking_content_buffer += chunk_content
                    with think_expander_placeholder.container():
                        with st.expander("CAG's thoughts...", expanded=True):
                            st.markdown(thinking_content_buffer + "▌")
                elif chunk_type == ChunkType.END_THINK:
                    with think_expander_placeholder.container():
                        with st.expander("CAG's thoughts...", expanded=True):
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