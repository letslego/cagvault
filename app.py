import os
import re
import json
import time
import streamlit as st
from config import Config, ModelConfig, ModelProvider
from config import (
    QWEN_3, DEEPSEEK_V3, DEEPSEEK_R1, MISTRAL_LARGE, MISTRAL_SMALL,
    LLAMA_3_3_70B, LLAMA_3_1_8B, PHI_4, GEMMA_2_27B, COMMAND_R_PLUS
)
from knowledge import KnowledgeType, KnowledgeSource, load_from_file, load_from_url
from models import create_llm, BaseChatModel
from chatbot import ChunkType, ask, create_context_cache, clear_cache
from kvcache import get_kv_cache
from qa_cache import get_qa_cache
from question_library import get_question_library

# Import Claude Skills PDF parser
from skills.pdf_parser.enhanced_parser import get_enhanced_parser
from skills.pdf_parser.ner_search import get_searchable_parser
from skills.mcp.claude_mcp_server import get_claude_mcp_server

st.set_page_config(
    page_title="CAG", layout="centered", initial_sidebar_state="expanded", page_icon="🌵"
)
 
st.header("🌵 CAG Agentic Chat")
st.subheader("Completely local and private chat with your documents")

# Model configurations with metadata
AVAILABLE_MODELS = {
    "Qwen3-14B (Default, 16GB)": {"config": QWEN_3, "ram": "16GB", "speed": "Fast"},
    "DeepSeek V3 (Best Quality, 64GB+)": {"config": DEEPSEEK_V3, "ram": "64GB+", "speed": "Slow"},
    "DeepSeek R1 (Reasoning, 32GB+)": {"config": DEEPSEEK_R1, "ram": "32GB+", "speed": "Medium"},
    "Mistral Large (Long Context, 32GB+)": {"config": MISTRAL_LARGE, "ram": "32GB+", "speed": "Medium"},
    "Mistral Small (Fast, 8GB)": {"config": MISTRAL_SMALL, "ram": "8GB", "speed": "Very Fast"},
    "Llama 3.3 70B (Strong Reasoning, 32GB+)": {"config": LLAMA_3_3_70B, "ram": "32GB+", "speed": "Medium"},
    "Llama 3.1 8B (Lightweight, 8GB)": {"config": LLAMA_3_1_8B, "ram": "8GB", "speed": "Very Fast"},
    "Phi-4 (Efficient, 16GB)": {"config": PHI_4, "ram": "16GB", "speed": "Fast"},
    "Gemma 2 27B (Balanced, 16GB+)": {"config": GEMMA_2_27B, "ram": "16GB+", "speed": "Fast"},
    "Command R+ (RAG-Optimized, 32GB+)": {"config": COMMAND_R_PLUS, "ram": "32GB+", "speed": "Medium"},
}

def get_current_model() -> ModelConfig:
    """Get the currently selected model from session state or config."""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Qwen3-14B (Default, 16GB)"
    return AVAILABLE_MODELS[st.session_state.selected_model]["config"]

@st.cache_resource(show_spinner=False)
def create_chatbot(_model_config: ModelConfig) -> BaseChatModel:
    """Create chatbot with specified model. Use _model_config to prevent hashing issues."""
    return create_llm(_model_config)

@st.cache_resource(show_spinner=False)
def init_pdf_parser():
    """Initialize enhanced PDF parser with NER and search."""
    return get_enhanced_parser()

@st.cache_resource(show_spinner=False)
def init_search_parser():
    """Initialize searchable parser."""
    return get_searchable_parser()


@st.cache_resource(show_spinner=False)
def init_mcp_server():
    """Initialize Claude MCP server for async tools."""
    return get_claude_mcp_server()

def load_sections_from_redis(doc_id: str):
    """Load sections for a document from Redis into parser memory."""
    if not doc_id:
        return []
    parser = init_pdf_parser()
    parser.load_document_from_redis(doc_id)
    return parser.memory.get_document_sections(doc_id)

def find_referenced_sections(text: str, sections):
    """Match sections by title or numbered prefix mentioned in the assistant reply."""

    def _candidates(title: str):
        """Generate matchable strings for a section title (case-insensitive)."""
        if not title:
            return []
        lowered = title.lower()
        cands = {lowered}
        # Capture numeric or roman prefixes like "5.12.2" or "IV."
        m = re.match(r"^([0-9ivxlcmIVXLCM]+(?:\.[0-9ivxlcmIVXLCM]+)*)", title.strip())
        if m:
            prefix = m.group(1).lower()
            cands.add(prefix)
            cands.add(f"section {prefix}")
            cands.add(f"§ {prefix}")
        return cands

    text_lower = text.lower()
    matched = []
    seen = set()

    for section in sections:
        sid = section.metadata.id
        if sid in seen:
            continue

        # Check main section
        for cand in _candidates(section.metadata.title):
            if cand and cand in text_lower:
                matched.append(section)
                seen.add(sid)
                break

        # Check subsections
        for sub in section.subsections:
            sub_sid = sub.metadata.id
            if sub_sid in seen:
                continue
            for cand in _candidates(sub.metadata.title):
                if cand and cand in text_lower:
                    matched.append(sub)
                    seen.add(sub_sid)
                    break

        if len(matched) >= 10:
            break

    return matched

def hydrate_source_from_redis(doc_meta: dict):
    """Create a KnowledgeSource from Redis-stored sections for chat context."""
    doc_id = doc_meta.get("document_id")
    if not doc_id:
        return None
    parser = init_pdf_parser()
    parser.load_document_from_redis(doc_id)
    sections = parser.memory.get_document_sections(doc_id)
    if not sections:
        return None
    # Concatenate sections into markdown with headings
    parts = []
    for sec in sections:
        meta = sec.metadata
        heading = "#" * max(1, meta.level)
        parts.append(f"{heading} {meta.title}\n\n{sec.content}\n")
    content = "\n".join(parts)
    return KnowledgeSource(
        id=f"redis_{doc_id}",
        name=doc_meta.get("document_name", doc_id),
        type=KnowledgeType.DOCUMENT,
        content=content
    )
 
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
if "mcp_tasks" not in st.session_state:
    st.session_state.mcp_tasks = {}
    
with st.sidebar:
    st.title("CAG Vault")
    
    # Model Selection
    with st.expander("🤖 **Model Settings**", expanded=False):
        st.markdown("**Select LLM Model**")
        
        current_model = st.session_state.get("selected_model", "Qwen3-14B (Default, 16GB)")
        
        selected_model = st.selectbox(
            "Choose model:",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(current_model),
            key="model_selector",
            help="Select the LLM model for chat. Restart required after changing."
        )
        
        # Show model info
        model_info = AVAILABLE_MODELS[selected_model]
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"**RAM:** {model_info['ram']}")
        with col2:
            st.caption(f"**Speed:** {model_info['speed']}")
        
        # Check if model changed
        if selected_model != st.session_state.get("selected_model"):
            st.session_state.selected_model = selected_model
            st.warning("⚠️ Model changed! Please restart the app to apply.")
            if st.button("🔄 Restart App", key="restart_app"):
                st.cache_resource.clear()
                st.rerun()
        
        # Show current model
        current_config = get_current_model()
        st.caption(f"**Active:** {current_config.name}")
        
        # Download script info
        st.markdown("---")
        st.markdown("**Download Models:**")
        st.code("./download_models.sh", language="bash")
        st.caption("Run this script to download all supported models")

    # Claude MCP async tools
    with st.expander("🔌 Claude MCP Tools", expanded=False):
        mcp_server = init_mcp_server()
        action = st.selectbox(
            "Pick a tool",
            options=["Web search", "Web search + API call"],
            key="mcp_action",
        )
        query = st.text_area("Query", placeholder="What do you want to research?", key="mcp_query")
        max_results = st.slider("Max results", min_value=3, max_value=15, value=5, step=1, key="mcp_max_results")

        followup_api = None
        payload_obj = None
        api_url = ""
        api_method = "POST"
        api_payload_text = ""

        if action == "Web search + API call":
            api_url = st.text_input("API URL", placeholder="https://example.com/hook", key="mcp_api_url")
            api_method = st.selectbox("HTTP method", ["POST", "GET"], key="mcp_api_method")
            api_payload_text = st.text_area("API payload (JSON)", placeholder='{"query": "..."}', key="mcp_api_payload")
            if api_payload_text:
                try:
                    payload_obj = json.loads(api_payload_text)
                except json.JSONDecodeError:
                    st.error("Payload must be valid JSON")
            followup_api = {"url": api_url.strip(), "payload": payload_obj, "method": api_method}

        if st.button("Run Claude MCP task", key="run_mcp_task"):
            if not query.strip():
                st.warning("Please enter a query")
            elif action == "Web search + API call" and not api_url.strip():
                st.warning("API URL required for follow-up call")
            elif action == "Web search + API call" and api_payload_text and payload_obj is None:
                st.warning("Fix JSON payload before running")
            else:
                task_id = mcp_server.start_web_search(query.strip(), max_results, followup_api if action == "Web search + API call" else None)
                st.session_state.mcp_tasks[task_id] = {"mode": action, "query": query.strip()}
                st.success(f"Started task {task_id[:8]}")
                st.rerun()

        if st.session_state.mcp_tasks:
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.markdown("**Task status**")
            with col2:
                if st.button("🔄 Check Status", key="refresh_tasks"):
                    st.rerun()
            to_remove = []
            for task_id, meta in list(st.session_state.mcp_tasks.items()):
                task = mcp_server.get_task(task_id)
                if not task:
                    continue
                status = task.status
                elapsed = time.time() - task.started_at
                st.caption(f"🔍 {meta['query'][:50]}...")
                if status in {"pending", "running"}:
                    st.info(f"⏳ Running... ({elapsed:.1f}s)")
                elif status == "done":
                    st.success(f"✅ Complete ({task.completed_at - task.started_at:.1f}s)")
                    with st.expander(f"Results", expanded=True):
                        if task.result:
                            res = task.result.get("search_results", {})
                            count = res.get('count', 0)
                            
                            if count > 0:
                                st.write(f"**Found {count} results**")
                                for item in res.get("items", [])[:5]:
                                    st.markdown(f"**[{item.get('title', 'No title')}]({item.get('url', '#')})**")
                                    st.caption(item.get('snippet', ''))
                                    st.markdown("---")
                            else:
                                st.info("DuckDuckGo returned no related topics. Showing abstract if available:")
                                raw = res.get("raw_data", {})
                                if raw.get("Abstract"):
                                    st.markdown(f"**{raw.get('Heading', 'Info')}**")
                                    st.write(raw.get("Abstract"))
                                    if raw.get("AbstractURL"):
                                        st.markdown(f"[Read more]({raw.get('AbstractURL')})")
                                else:
                                    st.warning("No results found. Try a different query or more specific terms.")
                            
                            if task.result.get("followup_api"):
                                st.markdown("**API Response:**")
                                st.json(task.result["followup_api"])
                    if st.button("✕ Dismiss", key=f"dismiss_{task_id}"):
                        to_remove.append(task_id)
                elif status == "error":
                    st.error(f"❌ {task.error or 'Task failed'}")
                    if st.button("✕ Dismiss", key=f"dismiss_{task_id}"):
                        to_remove.append(task_id)
                st.markdown("---")
            for task_id in to_remove:
                st.session_state.mcp_tasks.pop(task_id, None)
            if to_remove:
                st.rerun()
    
    st.markdown("---")
    
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
    
    # Question Library Management
    with st.expander("📚 **Question Library**", expanded=False):
        question_library = get_question_library()
        
        col1, col2 = st.columns(2)
        with col1:
            all_questions = question_library.get_all_questions()
            st.metric("Stored Questions", len(all_questions))
        with col2:
            if st.session_state.parsed_documents:
                doc_id = list(st.session_state.parsed_documents.values())[0].get("extraction_result", {}).get("document_id")
                if doc_id:
                    doc_questions = question_library.get_all_questions(doc_id)
                    st.metric("Doc Questions", len(doc_questions))
        
        st.markdown("**Search & Manage Questions:**")
        search_query = st.text_input("Search questions", placeholder="Type to search...", key="lib_search")
        
        if search_query:
            suggestions = question_library.get_autocomplete_suggestions(search_query, max_results=20)
            if suggestions:
                for qa in suggestions:
                    col1, col2 = st.columns([0.85, 0.15])
                    with col1:
                        st.caption(f"📌 {qa['question']}")
                        st.caption(f"Category: {qa['category_label']}")
                        usage = qa['metadata'].get('usage_count', 0)
                        st.caption(f"Uses: {usage}")
                    with col2:
                        if st.button("🗑️", key=f"del_q_{qa['question']}", help="Delete"):
                            question_library.remove_question(qa['question'])
                            st.rerun()
        else:
            # Show popular questions
            st.markdown("**Most Popular Questions:**")
            popular = question_library.get_popular_questions(limit=5)
            if popular:
                for qa in popular:
                    usage = qa['metadata'].get('usage_count', 0)
                    st.caption(f"📌 {qa['question']} (used {usage} times)")
            else:
                st.info("No questions in library yet")
        
        # Clear library option
        st.markdown("---")
        if st.button("🗑️ Clear All Questions"):
            question_library.clear_library()
            st.success("Cleared question library")
            st.rerun()
    
    st.markdown("---")
    
    # Q&A Cache Management
    with st.expander("💾 **Q&A Cache Management**", expanded=False):
        qa_cache = get_qa_cache()
        cache_stats = qa_cache.get_cache_stats()
        
        if cache_stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cached Q&A", cache_stats.get("total_qa_pairs", 0))
            with col2:
                st.metric("Docs Indexed", cache_stats.get("total_documents_indexed", 0))
            with col3:
                st.metric("Memory", cache_stats.get("redis_memory_usage", "N/A"))
        
        # View cached Q&A for each document (deduplicated by doc_id)
        if st.session_state.parsed_documents:
            st.markdown("**View cached questions by document:**")
            
            # Deduplicate documents by doc_id to avoid duplicate keys
            seen_docs = {}
            for source_id, parsed_info in st.session_state.parsed_documents.items():
                doc_id = parsed_info.get("extraction_result", {}).get("document_id")
                doc_name = parsed_info.get("document_name", "Unknown")
                if doc_id and doc_id not in seen_docs:
                    seen_docs[doc_id] = (source_id, doc_name)
            
            # Display each unique document
            for doc_id, (source_id, doc_name) in seen_docs.items():
                with st.expander(f"📄 {doc_name}", expanded=False):
                    qa_history = qa_cache.get_doc_qa_history(doc_id)
                    
                    if qa_history:
                        for i, qa in enumerate(qa_history, 1):
                            st.markdown(f"**Q{i}:** {qa.get('question')}")
                            with st.expander(f"Answer & Thinking", expanded=False):
                                if qa.get('thinking'):
                                    st.caption("💭 Thinking:")
                                    st.markdown(qa.get('thinking'))
                                st.caption("📝 Response:")
                                st.markdown(qa.get('response'))
                            st.caption(f"⏰ {qa.get('timestamp', 'N/A')[:10]}")
                    else:
                        st.info("No cached questions for this document yet")
                    
                    if st.button(f"🗑️ Clear cache for {doc_name}", key=f"clear_qa_{doc_id}"):
                        qa_cache.clear_doc_cache(doc_id)
                        st.success(f"Cleared cache for {doc_name}")
                        st.rerun()
        
        # Clear all Q&A cache
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Q&A Cache"):
                qa_cache.clear_all_cache()
                st.success("Cleared all Q&A cache")
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
            pdf_parser = init_pdf_parser()
            search_parser = init_search_parser()
            
            with st.spinner("📄 Converting PDF to text..."):
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

    # Redis document picker
    redis_docs = init_pdf_parser().list_redis_documents()
    with st.expander("🗄️ Documents in Redis", expanded=False):
        if not redis_docs:
            redis_url = os.getenv("REDIS_URL", "(not set)")
            st.info(f"No documents found in Redis. REDIS_URL={redis_url}")
            if st.button("🔄 Refresh Redis list", key="refresh_redis_docs"):
                st.rerun()
        else:
            options = {f"{d.get('document_name', 'doc')} ({d.get('document_id','')[:8]})": d for d in redis_docs}
            selection = st.selectbox(
                "Select a cached document to chat",
                options=list(options.keys()),
                key="redis_doc_select"
            ) if options else None
            if selection:
                chosen_meta = options[selection]
                if st.button("Load for chat", key="load_redis_doc"):
                    source = hydrate_source_from_redis(chosen_meta)
                    if source:
                        st.session_state.sources[source.id] = source
                        st.session_state.message_source_ids = {source.id}
                        st.session_state.context_cache_id = None
                        # Track parsed metadata for UI tabs
                        doc_id = chosen_meta.get("document_id")
                        st.session_state.parsed_documents[source.id] = {
                            'document_name': chosen_meta.get('document_name', source.name),
                            'document_id': doc_id,
                            'sections_count': len(load_sections_from_redis(doc_id)),
                            'extraction_result': {
                                'document_id': doc_id,
                                'document_name': chosen_meta.get('document_name'),
                                'pages': chosen_meta.get('pages', 0)
                            }
                        }
                        st.success(f"Loaded {chosen_meta.get('document_name')} from Redis for chat")
                        st.rerun()
                    else:
                        st.error("Could not load sections from Redis for this document.")
 
    with st.form(key="add_url_form", clear_on_submit=True):
        url_input = st.text_input(
            "Or, paste a Web Page Link (URL)",
            placeholder="e.g., https://test-url.com",
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
                        if doc_id:
                            parser.load_document_from_redis(doc_id)
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
                        search_method = st.radio(
                            "Search method",
                            ["Agentic (Claude-powered)", "Keyword", "Semantic"],
                            horizontal=True,
                            key=f"search_method_{source_id}"
                        )
                        
                        if search_query:
                            pdf_parser = init_pdf_parser()
                            pdf_parser.load_document_from_redis(doc_id)
                            sections = load_sections_from_redis(doc_id)
                            section_map = {s.metadata.id: s for s in sections}
                            
                            if search_method == "Agentic (Claude-powered)":
                                st.subheader("🤖 Agentic Search Results")
                                agentic_result = pdf_parser.search_parser.agentic_search(search_query, top_k=5)
                                if agentic_result.get('reasoning'):
                                    st.info(f"**Agent reasoning:** {agentic_result['reasoning']}")
                                for res in agentic_result.get("results", []):
                                    sid = res.get("section_id")
                                    section = section_map.get(sid)
                                    if not section:
                                        continue
                                    meta = section.metadata
                                    page_display = f"p. {meta.start_page}-{meta.end_page}" if (meta.start_page and meta.end_page) else f"p. {meta.page_range or meta.page_estimate}"
                                    relevance = res.get("relevance_explanation", "")
                                    with st.expander(f"{meta.title} ({page_display})"):
                                        st.caption(f"Why relevant: {relevance}")
                                        st.markdown(section.content)
                            
                            elif search_method == "Keyword":
                                st.subheader("🔍 Keyword Search Results")
                                results = pdf_parser.search_document(doc_id, search_query)
                                st.caption(f"Found {len(results.get('results', []))} sections")
                                for res in results.get("results", []):
                                    sid = res.get("section_id")
                                    section = section_map.get(sid)
                                    if not section:
                                        continue
                                    meta = section.metadata
                                    page_display = f"p. {meta.start_page}-{meta.end_page}" if (meta.start_page and meta.end_page) else f"p. {meta.page_range or meta.page_estimate}"
                                    with st.expander(f"{meta.title} ({page_display})"):
                                        st.markdown(section.content)
                                        if res.get("matches"):
                                            st.caption(f"Matches: {res.get('matches')}")
                            
                            else:  # Semantic
                                st.subheader("🧠 Semantic Search Results")
                                semantic_results = pdf_parser.search_parser.semantic_search(search_query, top_k=5)
                                if semantic_results:
                                    st.caption(f"Semantically similar sections")
                                    for res in semantic_results:
                                        sid = res.get("section_id")
                                        score = res.get("relevance_score", 0)
                                        section = section_map.get(sid)
                                        if not section:
                                            continue
                                        meta = section.metadata
                                        page_display = f"p. {meta.start_page}-{meta.end_page}" if (meta.start_page and meta.end_page) else f"p. {meta.page_range or meta.page_estimate}"
                                        with st.expander(f"{meta.title} ({page_display}) — Similarity: {score:.2%}"):
                                            st.markdown(section.content)
                                else:
                                    st.info("No semantically similar sections found.")
                    
                    with entities_tab:
                        entity_type_filter = st.selectbox(
                            "Filter by entity type",
                            ["All", "MONEY", "DATE", "PARTY", "AGREEMENT", "PERCENTAGE"],
                            key=f"entity_filter_{source_id}"
                        )
                        pdf_parser = init_pdf_parser()
                        pdf_parser.load_document_from_redis(doc_id)
                        entities = pdf_parser.search_parser.search_engine.search_entities(
                            doc_id,
                            None if entity_type_filter == "All" else entity_type_filter
                        )
                        sections = load_sections_from_redis(doc_id)
                        section_map = {s.metadata.id: s for s in sections}
                        if entities:
                            st.write(f"**Found {len(entities)} {entity_type_filter} entities**")
                            for entity in entities[:25]:
                                section = section_map.get(entity.section_id)
                                title = section.metadata.title if section else "Section"
                                with st.expander(f"{entity.type}: {entity.text} — {title}"):
                                    if section:
                                        st.markdown(section.content)
                                    else:
                                        st.caption(f"Section ID: {entity.section_id}")
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
            # Link section references in assistant replies to full sections from Redis
            if message["role"] == "assistant" and st.session_state.parsed_documents:
                # Collect sections for the current message's documents
                all_sections = []
                for source_id, parsed_info in st.session_state.parsed_documents.items():
                    doc_id = parsed_info.get("extraction_result", {}).get("document_id")
                    if not doc_id:
                        continue
                    all_sections.extend(load_sections_from_redis(doc_id))
                if all_sections:
                    matched = find_referenced_sections(message["content"], all_sections)
                    if matched:
                        st.markdown("**Referenced sections (click to expand)**")
                        for section in matched:
                            meta = section.metadata
                            page_display = f"p. {meta.start_page}-{meta.end_page}" if (meta.start_page and meta.end_page) else f"p. {meta.page_range or meta.page_estimate}"
                            with st.expander(f"🔗 {meta.title} ({page_display})"):
                                st.markdown(section.content)
                    else:
                        st.caption("No section titles detected in the answer — showing nothing to expand.")
            
    chat_enabled = len(st.session_state.sources) > 0
    
    # Question library and autocomplete suggestions
    if chat_enabled:
        question_library = get_question_library()
        
        # Get first document ID for personalized suggestions
        first_doc_id = None
        for source_id, parsed_info in st.session_state.parsed_documents.items():
            doc_id = parsed_info.get("extraction_result", {}).get("document_id")
            if doc_id:
                first_doc_id = doc_id
                break
        
        # Show popular/suggested questions as buttons
        with st.expander("💡 Suggested Questions", expanded=False):
            popular = question_library.get_popular_questions(doc_id=first_doc_id, limit=6)
            if popular:
                cols = st.columns(2)
                for idx, qa in enumerate(popular):
                    col = cols[idx % 2]
                    with col:
                        if st.button(
                            f"📌 {qa['question'][:50]}...",
                            key=f"suggested_{idx}",
                            use_container_width=True
                        ):
                            # Append to messages directly when button clicked
                            st.session_state.messages.append({"role": "user", "content": qa['question'], "thinking": None})
                            st.session_state.message_source_ids = set(st.session_state.sources.keys())
                            st.rerun()
            else:
                st.info("No suggested questions yet. Ask some questions to build the library!")
        
        # Show questions by category
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Browse by Category", use_container_width=True):
                st.session_state.show_category_browser = not st.session_state.get("show_category_browser", False)
        
        if st.session_state.get("show_category_browser", False):
            st.markdown("**Questions by Category:**")
            category_cols = st.columns(3)
            categories = list(question_library.CATEGORIES.items())
            
            for idx, (cat_key, cat_label) in enumerate(categories):
                col = category_cols[idx % 3]
                with col:
                    questions = question_library.get_questions_by_category(cat_key, doc_id=first_doc_id, limit=3)
                    if questions:
                        with st.expander(f"🏷️ {cat_label}", expanded=False):
                            for q in questions:
                                if st.button(q['question'], key=f"cat_{cat_key}_{q['question']}", use_container_width=True):
                                    st.session_state.messages.append({"role": "user", "content": q['question'], "thinking": None})
                                    st.session_state.message_source_ids = set(st.session_state.sources.keys())
                                    st.rerun()
    
    if prompt := st.chat_input(
        "Ask CAG about your documents...",
        key="chat_input",
        disabled=not chat_enabled,
    ):
        # Clear the input value state
        if "chat_input_value" in st.session_state:
            del st.session_state.chat_input_value
        
        st.session_state.messages.append({"role": "user", "content": prompt, "thinking": None})
        # Update message source IDs to current documents
        st.session_state.message_source_ids = set(st.session_state.sources.keys())
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant", avatar="🤖"):
            think_expander_placeholder = st.empty()
            message_placeholder = st.empty()
            cache_status_placeholder = st.empty()
            
            # Get current model and create chatbot
            current_model_config = get_current_model()
            llm = create_chatbot(current_model_config)
            query = st.session_state.messages[-1]["content"]
            history = st.session_state.messages[:-1]
            
            # Get document IDs for this query
            doc_ids = set()
            for source_id, parsed_info in st.session_state.parsed_documents.items():
                doc_id = parsed_info.get("extraction_result", {}).get("document_id")
                if doc_id:
                    doc_ids.add(doc_id)
            
            # Check Q&A cache
            qa_cache = get_qa_cache()
            question_library = get_question_library()
            cached_result = qa_cache.get_cached_qa(query, doc_ids) if doc_ids else None
            
            # Add question to library and increment usage
            if doc_ids:
                question_library.add_question(query, doc_ids=doc_ids)
                question_library.increment_usage(query)
            
            if cached_result:
                # Use cached response
                cache_status_placeholder.info("💾 Using cached response")
                
                thinking_content = cached_result.get("thinking")
                if thinking_content:
                    with think_expander_placeholder.container():
                        with st.expander("CAG's thoughts (cached)", expanded=False):
                            st.markdown(thinking_content)
                
                full_response_content = cached_result.get("response", "")
                message_placeholder.markdown(full_response_content)
            else:
                # Generate new response and cache it
                full_response_content = ""
                thinking_content_buffer = ""
                
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
                cache_status_placeholder.empty()
                
                # Cache the response
                if doc_ids:
                    qa_cache.cache_qa(
                        question=query,
                        response=full_response_content,
                        thinking=thinking_content_buffer if thinking_content_buffer else None,
                        doc_ids=doc_ids,
                        metadata={
                            "model": Config.MODEL,
                            "source_count": len(filtered_sources),
                            "has_thinking": bool(thinking_content_buffer)
                        }
                    )
            
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response_content,
                    "thinking": cached_result.get("thinking") if cached_result else thinking_content_buffer,
                }
            )
            st.rerun()