import os
import re
import json
import time
from datetime import datetime
import streamlit as st
from config import Config, ModelConfig, ModelProvider
from config import (
    QWEN_3, DEEPSEEK_V3, DEEPSEEK_R1, MISTRAL_LARGE, MISTRAL_SMALL,
    LLAMA_3_3_70B, LLAMA_3_1_8B, PHI_4, GEMMA_2_27B, COMMAND_R_PLUS
)
from knowledge import KnowledgeType, KnowledgeSource, load_from_file, load_from_url
from models import create_llm, BaseChatModel
from chatbot import ChunkType, ask
from qa_cache import get_qa_cache
from question_library import get_question_library
from lancedb_chat import create_lancedb_chat, FilterConfig
from lancedb_cache import get_lancedb_store
from agentic_rag import create_agentic_rag, AgenticRAG
from agent_sdk_tools import create_cag_mcp_server, CAG_TOOL_NAMES
from voice_features import get_voice_processor, VoiceConfig, is_voice_available
from data_lineage import get_lineage_tracker

# Import Claude Skills PDF parser
from skills.pdf_parser.enhanced_parser import get_enhanced_parser, Section, SectionMetadata
from skills.pdf_parser.ner_search import get_searchable_parser
from skills.mcp.claude_mcp_server import get_claude_mcp_server

# Agent SDK tool metadata (kept in sync with CAG_TOOL_NAMES)
AGENT_SDK_TOOL_DETAILS = {
    "mcp__cag-tools__web_search": {
        "label": "🌐 Web Search",
        "description": "Search the web for relevant information"
    },
    "mcp__cag-tools__extract_entities": {
        "label": "🔍 Extract Entities",
        "description": "Extract and categorize named entities from documents"
    },
    "mcp__cag-tools__rank_sections": {
        "label": "📊 Rank Sections",
        "description": "Rank document sections by relevance to a query"
    },
    "mcp__cag-tools__find_cross_references": {
        "label": "🔗 Find Cross References",
        "description": "Identify cross-references between document sections"
    },
    "mcp__cag-tools__verify_facts": {
        "label": "✓ Verify Facts",
        "description": "Verify factual claims against source documents"
    },
    "mcp__cag-tools__suggest_followups": {
        "label": "💡 Suggest Follow-ups",
        "description": "Generate relevant follow-up questions"
    },
}

# Knowledge base skills metadata (single source of truth for UI + inference)
KNOWLEDGE_BASE_SKILLS = {
    "analyze-financial-covenant": {
        "label": "📊 Financial Covenants",
        "description": "Leverage ratios, coverage ratios, EBITDA adjustments, equity cure"
    },
    "analyze-default-provisions": {
        "label": "⚠️ Default Provisions",
        "description": "Events of default, grace periods, cure rights, cross-defaults"
    },
    "analyze-events-of-default": {
        "label": "🚨 Events of Default",
        "description": "Default clauses, thresholds, cross-acceleration, remedies"
    },
    "extract-defined-terms": {
        "label": "📖 Defined Terms",
        "description": "Term extraction, nested definitions, carve-outs, baskets"
    },
    "compare-market-standard": {
        "label": "🎯 Market Standards",
        "description": "LMA/LSTA comparison, borrower vs lender-favorable terms"
    },
    "analyze-nsl-provisions": {
        "label": "🔒 Net Short Lender",
        "description": "NSL definitions, voting restrictions, transfer limitations"
    },
    "analyze-debt-baskets": {
        "label": "💰 Debt Baskets",
        "description": "Ratio-based debt, general baskets, Available Amount"
    },
    "analyze-negative-covenants": {
        "label": "⛔ Negative Covenants",
        "description": "Restricted payments, asset sales, liens, investments"
    },
    "analyze-asset-sales-dispositions": {
        "label": "🏢 Asset Sales",
        "description": "Permitted dispositions, reinvestment, excess proceeds"
    },
    "analyze-prepayments": {
        "label": "💸 Prepayments",
        "description": "Mandatory/optional prepayments, make-whole provisions"
    },
    "analyze-lender-protections": {
        "label": "🛡️ Lender Protections",
        "description": "Sharing clauses, set-off rights, indemnities"
    },
    "analyze-material-adverse-effect": {
        "label": "⚡ MAE/MAC",
        "description": "MAE definitions, carve-outs, burden of proof"
    },
}

# Lightweight heuristics to infer which skills to apply for a question
SKILL_KEYWORDS = {
    "analyze-financial-covenant": ["covenant", "leverage", "coverage", "ebitda", "ratio", "financial covenant"],
    "analyze-default-provisions": ["default", "cure", "remedy", "cross-default", "acceleration"],
    "analyze-events-of-default": ["event of default", "eod", "acceleration", "termination"],
    "extract-defined-terms": ["define", "definition", "means", "defined term"],
    "compare-market-standard": ["market", "lma", "lsta", "market standard", "borrower-friendly", "lender-friendly"],
    "analyze-nsl-provisions": ["net short", "nsl"],
    "analyze-debt-baskets": ["debt basket", "incurrence", "available amount", "ratio debt", "basket"],
    "analyze-negative-covenants": ["negative covenant", "restricted payment", "rp", "lien", "investment", "affiliate transaction"],
    "analyze-asset-sales-dispositions": ["asset sale", "disposition", "proceeds", "reinvestment", "sweep"],
    "analyze-prepayments": ["prepay", "prepayment", "call protection", "make-whole", "soft call"],
    "analyze-lender-protections": ["lender protection", "set-off", "setoff", "indemnity", "sharing"],
    "analyze-material-adverse-effect": ["mae", "mac", "material adverse"],
}

def infer_skills_from_question(question: str) -> list:
    """Infer relevant skills based on simple keyword matches."""
    q_lower = (question or "").lower()
    hits = []
    for skill_id, keywords in SKILL_KEYWORDS.items():
        if any(keyword in q_lower for keyword in keywords):
            hits.append(skill_id)
    # Fallback to defined terms when nothing matches
    return hits or ["extract-defined-terms"]

def render_skill_list(skills: list, label: str = "Skills used", expanded: bool = False):
    """Render a collapsible list of skills with descriptions."""
    if not skills:
        return
    with st.expander(f"📚 {label}", expanded=expanded):
        for skill_id in skills:
            meta = KNOWLEDGE_BASE_SKILLS.get(skill_id, {"label": skill_id, "description": ""})
            st.markdown(f"- **{meta['label']}** — {meta['description']}")
            st.caption(skill_id)

st.set_page_config(
    page_title="CAG", layout="centered", initial_sidebar_state="expanded", page_icon="🌵"
)

# Navigation
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.header("🌵 CAG Agentic Chat")
with col2:
    if st.button("📊 Lineage", help="View data pipeline lineage and metrics"):
        st.switch_page("pages/lineage_dashboard.py")

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

@st.cache_resource(show_spinner=False)
def init_agentic_rag(_model_config: ModelConfig) -> AgenticRAG:
    """Initialize Agentic RAG system with multi-step reasoning and Agent SDK MCP tools."""
    llm = create_llm(_model_config)
    parser = get_enhanced_parser()
    search_parser = get_searchable_parser()
    
    # Create Agent SDK MCP server with CAG tools
    try:
        mcp_server_config = create_cag_mcp_server()
        st.info(f"✅ Agent SDK MCP Server initialized with {len(CAG_TOOL_NAMES)} tools")
    except ImportError as e:
        st.warning(f"⚠️ Agent SDK not available: {e}. Continuing without MCP tools.")
        mcp_server_config = None
    
    return create_agentic_rag(
        llm, 
        parser, 
        search_parser, 
        enable_reflection=True,
        mcp_server_config=mcp_server_config
    )

@st.cache_resource(show_spinner=False)
def init_lancedb_chat():
    """Initialize LanceDB multi-document chat system."""
    current_model = get_current_model()
    search_mode = st.session_state.get("lancedb_search_mode", "semantic")
    reranker_type = st.session_state.get("lancedb_reranker_type", "linear")
    reranker_model = st.session_state.get("lancedb_reranker_model")

    return create_lancedb_chat(
        db_path="./lancedb",
        table_name="documents",
        llm_model=current_model.name,
        llm_base_url=Config.OLLAMA_BASE_URL,
        search_mode=search_mode,
        hybrid_weight=0.7,  # 70% semantic, 30% keyword
        reranker_type=reranker_type,
        reranker_model=reranker_model,
    )

def load_sections_from_lancedb(doc_id: str):
    """Load sections for a document from LanceDB into parser memory."""
    if not doc_id:
        return []
    parser = init_pdf_parser()
    parser.load_document_from_lancedb(doc_id)
    sections = parser.memory.get_document_sections(doc_id)
    if sections:
        return sections

    # Fallback: hydrate from LanceDB tables if parser memory is empty
    try:
        import json
        store = get_lancedb_store()
        rows = store.load_sections(doc_id)
        if not rows:
            return []
        parser.memory.clear_document(doc_id)
        for row in rows:
            meta_dict = json.loads(row.get("metadata_json", "{}")) if row.get("metadata_json") else {}
            title = meta_dict.get("title") or row.get("title", "Untitled")
            level = int(meta_dict.get("level", row.get("level", 1) or 1))
            content = row.get("content", "")
            meta = SectionMetadata(
                title=title,
                level=level,
                start_line=int(meta_dict.get("start_line", 0)),
                end_line=int(meta_dict.get("end_line", 0)),
                content_length=len(content),
                word_count=len(content.split()),
                has_code=bool(meta_dict.get("has_code", False)),
                has_tables=bool(meta_dict.get("has_tables", False)),
                subsection_count=int(meta_dict.get("subsection_count", 0)),
                parent_id=meta_dict.get("parent_id"),
                page_estimate=int(meta_dict.get("page_estimate", 1)),
                page_range=meta_dict.get("page_range"),
                start_page=int(meta_dict.get("start_page", 1)),
                end_page=int(meta_dict.get("end_page", 1)),
                section_type=meta_dict.get("section_type"),
                importance_score=float(meta_dict.get("importance_score", 0.5)),
                typical_dependencies=meta_dict.get("typical_dependencies", []),
            )
            sec = Section(metadata=meta, content=content, document_id=doc_id, subsections=[])
            parser.memory.add_section(doc_id, sec)
        return parser.memory.get_document_sections(doc_id)
    except Exception:
        return []

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

def hydrate_source_from_lancedb(doc_meta: dict):
    """Create a KnowledgeSource from LanceDB-stored sections for chat context."""
    doc_id = doc_meta.get("document_id")
    if not doc_id:
        return None
    sections = load_sections_from_lancedb(doc_id)
    if not sections:
        # Fallback: try raw rows to still build a source for chat
        try:
            import json
            store = get_lancedb_store()
            rows = store.load_sections(doc_id)
            if not rows:
                return None
            parts = []
            for row in rows:
                meta = json.loads(row.get("metadata_json", "{}")) if row.get("metadata_json") else {}
                title = meta.get("title") or row.get("title", "Untitled")
                level = int(meta.get("level", row.get("level", 1) or 1))
                heading = "#" * max(1, level)
                parts.append(f"{heading} {title}\n\n{row.get('content', '')}\n")
            content = "\n".join(parts)
            return KnowledgeSource(
                id=f"lancedb_{doc_id}",
                name=doc_meta.get("document_name", doc_id),
                type=KnowledgeType.DOCUMENT,
                content=content
            )
        except Exception:
            return None
    parts = []
    for sec in sections:
        meta = sec.metadata
        heading = "#" * max(1, meta.level)
        parts.append(f"{heading} {meta.title}\n\n{sec.content}\n")
    content = "\n".join(parts)
    return KnowledgeSource(
        id=f"lancedb_{doc_id}",
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
if "message_source_ids" not in st.session_state:
    st.session_state.message_source_ids = set()
if "parsed_documents" not in st.session_state:
    st.session_state.parsed_documents = {}  # Store parsed PDF metadata
if "mcp_tasks" not in st.session_state:
    st.session_state.mcp_tasks = {}
if "lancedb_documents" not in st.session_state:
    st.session_state.lancedb_documents = []
if "lancedb_indexed" not in st.session_state:
    st.session_state.lancedb_indexed = False
if "lancedb_chat_mode" not in st.session_state:
    st.session_state.lancedb_chat_mode = False
if "lancedb_search_mode" not in st.session_state:
    st.session_state.lancedb_search_mode = "hybrid"  # 'semantic' or 'hybrid'
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = is_voice_available()
if "voice_input_enabled" not in st.session_state:
    st.session_state.voice_input_enabled = False
if "voice_output_enabled" not in st.session_state:
    st.session_state.voice_output_enabled = False
    
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

    # Agent SDK MCP Tools
    with st.expander("🛠️ Agent SDK MCP Tools", expanded=False):
        st.markdown("### Available Tools")
        st.markdown("These tools are available via Claude Agent SDK for credit agreement analysis:")
        
        for tool_name in CAG_TOOL_NAMES:
            tool_meta = AGENT_SDK_TOOL_DETAILS.get(tool_name, None)
            label = tool_meta["label"] if tool_meta else tool_name
            description = tool_meta.get("description") if tool_meta else "Available Agent SDK tool"
            st.markdown(f"- **{label}** — {description}")
            st.caption(tool_name)
        
        st.markdown("---")
        st.info(f"ℹ️ Tools are automatically available when using chat with Agent SDK enabled. Total: {len(CAG_TOOL_NAMES)} tools")
    
    # Knowledge Base Skills
    with st.expander("📚 Knowledge Base Skills", expanded=False):
        st.markdown("### Credit Agreement Analysis Skills")
        st.markdown("Specialized frameworks for comprehensive credit agreement analysis:")
        
        for skill_id, meta in KNOWLEDGE_BASE_SKILLS.items():
            st.markdown(f"- **{meta['label']}** — {meta['description']}")
            st.caption(skill_id)
        
        st.markdown("---")
        st.info("ℹ️ Skills are automatically activated based on question type. Located in `knowledge-base/` folder.")

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
                            st.caption(f"Debug: count={count}, items={len(res.get('items', []))}, has_raw={bool(res.get('raw_data'))}")
                            
                            if count > 0:
                                items = res.get("items", [])
                                top = items[0]
                                # Show a quick inline answer from the first result
                                st.write(f"**Answer:** {top.get('snippet', top.get('title', ''))}")
                                st.write(f"Source: {top.get('title', 'No title')}")
                                st.markdown(f"[Open link]({top.get('url', '#')})")
                                st.markdown("---")
                                # Still list a few more links below
                                st.write(f"**Found {count} results**")
                                for item in items[:5]:
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
    
    # Show LanceDB QA cache stats
    qa_cache = get_qa_cache()
    qa_stats = qa_cache.get_stats()
    if qa_stats:
        with st.expander("📊 QA Cache (LanceDB)", expanded=False):
            st.metric("Cached Q&A Pairs", qa_stats.get("total_qa_pairs", 0))
            st.metric("Documents Indexed", qa_stats.get("total_documents_indexed", 0))
            if st.button("🧹 Clear QA Cache"):
                qa_cache.clear_all_cache()
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
        cache_stats = qa_cache.get_stats()
        
        if cache_stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cached Q&A", cache_stats.get("total_qa_pairs", 0))
            with col2:
                st.metric("Docs Indexed", cache_stats.get("total_documents_indexed", 0))
        
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
                    qa_history = qa_cache.get_doc_history(doc_id)
                    
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
    
    # Agentic RAG Mode
    with st.expander("🤖 **Agentic RAG (Advanced)**", expanded=False):
        st.markdown("**Multi-step reasoning with self-reflection**")
        
        agentic_rag_enabled = st.checkbox(
            "Enable Agentic RAG",
            value=st.session_state.get("agentic_rag_mode", False),
            key="agentic_rag_toggle",
            help="Advanced RAG with query understanding, strategy selection, and answer validation"
        )
        
        if agentic_rag_enabled != st.session_state.get("agentic_rag_mode", False):
            st.session_state.agentic_rag_mode = agentic_rag_enabled
            st.rerun()
        
        if agentic_rag_enabled:
            st.caption("✨ Features:")
            st.caption("• Query intent understanding")
            st.caption("• Automatic retrieval strategy selection")  
            st.caption("• Multi-source answer synthesis")
            st.caption("• Self-reflection & validation")
            
            # Configuration
            st.markdown("**Configuration**")
            
            if "agentic_reflection" not in st.session_state:
                st.session_state.agentic_reflection = True
            enable_reflection = st.checkbox(
                "Enable Self-Reflection",
                value=st.session_state.agentic_reflection,
                key="agentic_reflection",
                help="Validate answer quality before presenting"
            )
            
            if "agentic_max_context" not in st.session_state:
                st.session_state.agentic_max_context = 16000
            max_context = st.slider(
                "Max Context Length",
                min_value=4000,
                max_value=32000,
                value=st.session_state.agentic_max_context,
                step=1000,
                key="agentic_max_context",
                help="Maximum tokens of context to retrieve"
            )
            
            if "agentic_show_reasoning" not in st.session_state:
                st.session_state.agentic_show_reasoning = True
            show_reasoning = st.checkbox(
                "Show Reasoning Steps",
                value=st.session_state.agentic_show_reasoning,
                key="agentic_show_reasoning",
                help="Display agent's reasoning process"
            )
    
    st.markdown("---")
    
    # LanceDB Multi-Document Chat
    with st.expander("🗄️ **LanceDB Multi-Document Chat**", expanded=False):
        st.markdown("**Enable advanced multi-document search with LanceDB**")
        
        # Toggle for LanceDB mode
        lancedb_enabled = st.checkbox(
            "Enable LanceDB Chat",
            value=st.session_state.lancedb_chat_mode,
            key="lancedb_toggle",
            help="Use LanceDB for semantic/hybrid search across multiple documents"
        )
        
        if lancedb_enabled != st.session_state.lancedb_chat_mode:
            st.session_state.lancedb_chat_mode = lancedb_enabled
            st.rerun()
        
        if lancedb_enabled:
            # Search mode selector
            search_mode = st.radio(
                "Search Mode",
                ["Semantic", "Hybrid"],
                index=0 if st.session_state.lancedb_search_mode == "semantic" else 1,
                key="lancedb_search_mode_radio",
                help="Semantic: Vector similarity only | Hybrid: Vector + Keyword search",
                horizontal=True
            )
            
            new_mode = search_mode.lower()
            if new_mode != st.session_state.lancedb_search_mode:
                st.session_state.lancedb_search_mode = new_mode
                st.session_state.lancedb_indexed = False  # Require re-indexing
                st.cache_resource.clear()  # Clear cached LanceDB instance
                st.info(f"Switched to {search_mode} search. Please re-index documents.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents Indexed", len(st.session_state.lancedb_documents))
            with col2:
                indexed_status = "✅ Indexed" if st.session_state.lancedb_indexed else "⚠️ Not Indexed"
                st.caption(indexed_status)
            
            if st.session_state.lancedb_search_mode == "hybrid":
                st.caption("🔍 Hybrid search combines semantic similarity with keyword matching")
            
            # Filtering options
            st.markdown("---")
            st.markdown("**🔎 Search Filters** *(optional)*")
            
            # Initialize filter state
            if "lancedb_filter_enabled" not in st.session_state:
                st.session_state.lancedb_filter_enabled = False
            
            enable_filters = st.checkbox(
                "Enable Pre/Post Filtering",
                value=st.session_state.lancedb_filter_enabled,
                key="enable_filters_checkbox",
                help="Filter documents by file type, date, or other criteria"
            )
            st.session_state.lancedb_filter_enabled = enable_filters
            
            if enable_filters:
                # File type filter (pre-filter)
                file_types = list(set([doc.get("file_type", "pdf") for doc in st.session_state.lancedb_documents]))
                if file_types:
                    selected_types = st.multiselect(
                        "Filter by File Type (pre-filter)",
                        options=file_types,
                        default=file_types,
                        key="filter_file_type",
                        help="Applied before vector search for efficiency"
                    )
                    if selected_types:
                        types_filter = " OR ".join([f"metadata.file_type = '{t}'" for t in selected_types])
                        st.session_state.lancedb_pre_filter = types_filter if len(selected_types) < len(file_types) else None
                    else:
                        st.session_state.lancedb_pre_filter = None
                
                # Date filter (pre-filter)
                use_date_filter = st.checkbox("Filter by Upload Date", key="use_date_filter")
                if use_date_filter:
                    from datetime import datetime, timedelta
                    date_from = st.date_input(
                        "From Date",
                        value=datetime.now() - timedelta(days=7),
                        key="filter_date_from"
                    )
                    date_filter = f"metadata.upload_date >= '{date_from.isoformat()}'"
                    if "lancedb_pre_filter" in st.session_state and st.session_state.lancedb_pre_filter:
                        st.session_state.lancedb_pre_filter = f"({st.session_state.lancedb_pre_filter}) AND {date_filter}"
                    else:
                        st.session_state.lancedb_pre_filter = date_filter
                
                # File size filter (post-filter)
                use_size_filter = st.checkbox("Filter by File Size", key="use_size_filter")
                if use_size_filter:
                    min_size = st.number_input(
                        "Minimum Size (bytes)",
                        min_value=0,
                        value=0,
                        key="filter_min_size",
                        help="Applied after vector search for refinement"
                    )
                    if min_size > 0:
                        st.session_state.lancedb_post_filter = f"file_size >= {min_size}"
                    else:
                        st.session_state.lancedb_post_filter = None
                
                # Display active filters
                if st.session_state.get("lancedb_pre_filter") or st.session_state.get("lancedb_post_filter"):
                    st.caption("**Active Filters:**")
                    if st.session_state.get("lancedb_pre_filter"):
                        st.caption(f"⚡ Pre-filter: `{st.session_state.lancedb_pre_filter}`")
                    if st.session_state.get("lancedb_post_filter"):
                        st.caption(f"🎯 Post-filter: `{st.session_state.lancedb_post_filter}`")
            else:
                # Clear filters when disabled
                st.session_state.lancedb_pre_filter = None
                st.session_state.lancedb_post_filter = None
            
            st.markdown("---")
            
            # Advanced FTS Options (only for hybrid mode)
            if st.session_state.lancedb_search_mode == "hybrid":
                st.markdown("**⚡ Advanced Full-Text Search**")
                
                # Initialize advanced search state
                if "lancedb_fts_enabled" not in st.session_state:
                    st.session_state.lancedb_fts_enabled = False
                
                enable_advanced_fts = st.checkbox(
                    "Enable Advanced FTS",
                    value=st.session_state.lancedb_fts_enabled,
                    key="enable_advanced_fts",
                    help="Multi-match, boosting, phrase, fuzzy, and boolean queries"
                )
                st.session_state.lancedb_fts_enabled = enable_advanced_fts
                
                if enable_advanced_fts:
                    # Query type selector
                    fts_query_type = st.selectbox(
                        "FTS Query Type",
                        ["match", "multi_match", "phrase", "fuzzy", "boolean"],
                        index=0,
                        key="fts_query_type",
                        help="Match: Simple | Multi-match: Multiple fields | Phrase: Exact phrase | Fuzzy: Typo-tolerant | Boolean: AND/OR"
                    )
                    st.session_state.lancedb_fts_query_type = fts_query_type
                    
                    # Multi-match settings
                    if fts_query_type == "multi_match":
                        st.caption("**Multi-Match Configuration**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            search_text = st.checkbox("Search in Text", value=True, key="mm_text")
                            search_summary = st.checkbox("Search in Summary", value=True, key="mm_summary")
                        with col_b:
                            search_filename = st.checkbox("Search in Filename", value=True, key="mm_filename")
                        
                        # Field boosting
                        use_boosting = st.checkbox("Enable Field Boosting", key="enable_boosting")
                        if use_boosting:
                            st.caption("**Field Boost Weights** *(higher = more important)*")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                text_boost = st.number_input("Text", min_value=0.1, max_value=5.0, value=2.0, step=0.1, key="boost_text")
                            with col2:
                                summary_boost = st.number_input("Summary", min_value=0.1, max_value=5.0, value=1.5, step=0.1, key="boost_summary")
                            with col3:
                                filename_boost = st.number_input("Filename", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="boost_filename")
                            
                            st.session_state.lancedb_field_boosts = {
                                "text": text_boost if search_text else 0,
                                "metadata.summary": summary_boost if search_summary else 0,
                                "metadata.filename": filename_boost if search_filename else 0,
                            }
                        
                        # Build fields list
                        fields = []
                        if search_text:
                            fields.append("text")
                        if search_summary:
                            fields.append("metadata.summary")
                        if search_filename:
                            fields.append("metadata.filename")
                        st.session_state.lancedb_fts_fields = fields
                    
                    # Phrase query settings
                    elif fts_query_type == "phrase":
                        phrase_slop = st.slider(
                            "Phrase Slop",
                            min_value=0,
                            max_value=10,
                            value=0,
                            key="phrase_slop",
                            help="Allow N words between phrase terms (0 = exact match)"
                        )
                        st.session_state.lancedb_phrase_slop = phrase_slop
                        if phrase_slop > 0:
                            st.caption(f"💡 Will match phrases with up to {phrase_slop} word(s) between terms")
                    
                    # Fuzzy search settings
                    elif fts_query_type == "fuzzy":
                        fuzzy_distance = st.slider(
                            "Edit Distance",
                            min_value=1,
                            max_value=3,
                            value=2,
                            key="fuzzy_distance",
                            help="Maximum Levenshtein distance for fuzzy matching"
                        )
                        st.session_state.lancedb_fuzzy_distance = fuzzy_distance
                        st.caption(f"💡 Tolerates up to {fuzzy_distance} character edit(s) per word")
                    
                    # Boolean query settings
                    elif fts_query_type == "boolean":
                        boolean_op = st.radio(
                            "Boolean Operator",
                            ["AND", "OR"],
                            index=0,
                            key="boolean_operator",
                            help="AND: All terms must match | OR: Any term can match",
                            horizontal=True
                        )
                        st.session_state.lancedb_boolean_operator = boolean_op
                    
                    # Overall boost
                    overall_boost = st.slider(
                        "Overall Query Boost",
                        min_value=0.1,
                        max_value=3.0,
                        value=1.0,
                        step=0.1,
                        key="overall_boost",
                        help="Boost the entire query's score"
                    )
                    st.session_state.lancedb_overall_boost = overall_boost
                else:
                    # Clear advanced FTS settings
                    st.session_state.lancedb_fts_query_type = "match"
                    st.session_state.lancedb_fts_fields = None
                    st.session_state.lancedb_field_boosts = None
                
                st.markdown("---")
            
            # Reranking Configuration
            st.markdown("**🎯 Reranking Configuration**")
            
            # Initialize reranker state
            if "lancedb_reranker_type" not in st.session_state:
                st.session_state.lancedb_reranker_type = "linear"
            
            reranker_type = st.selectbox(
                "Reranker Type",
                ["linear", "cross_encoder", "colbert", "cohere"],
                index=["linear", "cross_encoder", "colbert", "cohere"].index(st.session_state.lancedb_reranker_type),
                key="reranker_type_select",
                help="Linear: Fast hybrid | Cross-encoder: High accuracy | ColBERT: Neural | Cohere: API-based"
            )
            
            if reranker_type != st.session_state.lancedb_reranker_type:
                st.session_state.lancedb_reranker_type = reranker_type
                st.cache_resource.clear()
                st.info(f"Switched to {reranker_type} reranker. Please re-index documents.")
            
            # Model selection for specific rerankers
            if reranker_type == "cross_encoder":
                model = st.text_input(
                    "Cross-Encoder Model",
                    value="BAAI/bge-reranker-base",
                    key="cross_encoder_model",
                    help="HuggingFace cross-encoder model"
                )
                st.session_state.lancedb_reranker_model = model
            elif reranker_type == "colbert":
                model = st.text_input(
                    "ColBERT Model",
                    value="colbert-ir/colbertv2.0",
                    key="colbert_model",
                    help="ColBERT model for neural reranking"
                )
                st.session_state.lancedb_reranker_model = model
            elif reranker_type == "cohere":
                model = st.text_input(
                    "Cohere Model",
                    value="rerank-english-v2.0",
                    key="cohere_model",
                    help="Cohere rerank model (requires API key)"
                )
                st.session_state.lancedb_reranker_model = model
                st.caption("⚠️ Requires COHERE_API_KEY environment variable")
            else:
                st.session_state.lancedb_reranker_model = None
            
            st.markdown("---")
            
            # Multivector Search Configuration
            st.markdown("**🔀 Multivector Search**")
            
            # Initialize multivector state
            if "lancedb_multivector_enabled" not in st.session_state:
                st.session_state.lancedb_multivector_enabled = False
            
            enable_multivector = st.checkbox(
                "Enable Multivector Search",
                value=st.session_state.lancedb_multivector_enabled,
                key="enable_multivector",
                help="Search with multiple query variations for better recall"
            )
            st.session_state.lancedb_multivector_enabled = enable_multivector
            
            if enable_multivector:
                st.caption("💡 Searches with multiple query phrasings and aggregates results")
                
                # Option for custom variations
                use_custom_variations = st.checkbox(
                    "Provide Custom Query Variations",
                    key="use_custom_variations",
                    help="Manually specify query variations instead of auto-generating"
                )
                
                if use_custom_variations:
                    variations = st.text_area(
                        "Query Variations (one per line)",
                        placeholder="alternative phrasing 1\nalternative phrasing 2",
                        key="multivector_variations",
                        help="Enter alternative phrasings of your query"
                    )
                    if variations:
                        st.session_state.lancedb_query_variations = [v.strip() for v in variations.split('\n') if v.strip()]
                    else:
                        st.session_state.lancedb_query_variations = None
                else:
                    st.session_state.lancedb_query_variations = None
                    st.caption("🤖 Auto-generating query variations using LLM")
            
            st.markdown("---")
            
            # Index documents button
            if st.session_state.lancedb_documents and not st.session_state.lancedb_indexed:
                if st.button("🔄 Index Documents", key="index_lancedb"):
                    try:
                        lancedb_chat = init_lancedb_chat()
                        with st.spinner("Indexing documents..."):
                            lancedb_chat.add_documents(st.session_state.lancedb_documents)
                        st.session_state.lancedb_indexed = True
                        st.success("Documents indexed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Indexing failed: {str(e)}")
            
            # Clear index button
            if st.session_state.lancedb_indexed:
                if st.button("🗑️ Clear Index", key="clear_lancedb"):
                    try:
                        lancedb_chat = init_lancedb_chat()
                        lancedb_chat.clear_documents()
                        st.session_state.lancedb_documents = []
                        st.session_state.lancedb_indexed = False
                        st.success("Index cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Clear failed: {str(e)}")
            
            st.caption("💡 Documents are automatically added when uploaded")
    
    st.markdown("---")
    
    # Voice Features
    if st.session_state.voice_enabled:
        with st.expander("🎤 **Voice Features**", expanded=False):
            st.markdown("**Speech-to-text and text-to-speech**")
            
            col1, col2 = st.columns(2)
            with col1:
                enable_voice_input = st.checkbox(
                    "🎙️ Voice Input",
                    value=st.session_state.voice_input_enabled,
                    key="voice_input_toggle",
                    help="Record questions via microphone"
                )
                st.session_state.voice_input_enabled = enable_voice_input
            
            with col2:
                enable_voice_output = st.checkbox(
                    "🔊 Voice Output",
                    value=st.session_state.voice_output_enabled,
                    key="voice_output_toggle",
                    help="Synthesize answers to speech"
                )
                st.session_state.voice_output_enabled = enable_voice_output
            
            if enable_voice_input or enable_voice_output:
                # Voice Configuration
                st.markdown("**Configuration**")
                
                if enable_voice_input:
                    st.caption("🎙️ **Voice Input Settings**")
                    record_duration = st.slider(
                        "Recording Duration (seconds)",
                        min_value=5,
                        max_value=60,
                        value=10,
                        step=5,
                        key="voice_record_duration",
                        help="Maximum recording time"
                    )
                
                if enable_voice_output:
                    st.caption("🔊 **Voice Output Settings**")
                    col1, col2 = st.columns(2)
                    with col1:
                        tts_rate = st.slider(
                            "Speech Rate (WPM)",
                            min_value=50,
                            max_value=300,
                            value=150,
                            step=10,
                            key="voice_tts_rate",
                            help="Words per minute"
                        )
                    
                    with col2:
                        tts_volume = st.slider(
                            "Volume",
                            min_value=0.0,
                            max_value=1.0,
                            value=1.0,
                            step=0.1,
                            key="voice_tts_volume"
                        )
    else:
        with st.expander("🎤 **Voice Features (Disabled)**", expanded=False):
            st.warning(
                "Voice features require additional dependencies. Install with:\n"
                "`pip install pyttsx3 sounddevice soundfile openai`"
            )
    
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
        
        # Track document ingestion
        lineage = get_lineage_tracker()
        start_time = time.time()
        
        try:
            # Track PDF ingestion
            doc_id = lineage.track_pdf_ingestion(uploaded_file.name, uploaded_file.size)
            
            # Load file into knowledge base
            source = load_from_file(uploaded_file.name, uploaded_file.read())
            st.session_state.sources[source.id] = source
            st.session_state.message_source_ids = {source.id}
            
            # Parse with Claude Skills (persists to LanceDB inside parser)
            pdf_parser = init_pdf_parser()
            search_parser = init_search_parser()
            
            with st.spinner("📄 Converting PDF to text..."):
                # Extract sections
                extraction_start = time.time()
                extraction_result = pdf_parser.parse_and_extract_sections(tmp_path)
                extraction_duration = (time.time() - extraction_start) * 1000
                
                # Track section extraction
                sections = extraction_result.get('sections', [])
                for i, section in enumerate(sections[:10]):  # Track first 10 sections
                    section_text = section.get('text', '') if isinstance(section, dict) else str(section)
                    lineage.track_section_extraction(
                        doc_id, 
                        f"section_{i}_{section.get('title', f'Section {i}')[:20]}" if isinstance(section, dict) else f"section_{i}",
                        section_text,
                        extraction_duration / len(sections) if sections else extraction_duration
                    )
                
                # Store parsed metadata
                doc_id_parsed = extraction_result.get('document_id')
                st.session_state.parsed_documents[source.id] = {
                    'document_name': uploaded_file.name,
                    'document_id': doc_id_parsed,
                    'sections_count': extraction_result.get('sections_count', 0),
                    'extraction_result': extraction_result
                }
                
                # Queue for LanceDB chat ingestion, deduped by document id
                file_type = uploaded_file.name.split(".")[-1].lower()
                file_size = uploaded_file.size
                already_queued = any(doc.get("document_id") == doc_id_parsed for doc in st.session_state.lancedb_documents)
                if not already_queued:
                    st.session_state.lancedb_documents.append({
                        "name": uploaded_file.name,
                        "document_id": doc_id_parsed,
                        "content": source.content,
                        "file_type": file_type,
                        "file_size": file_size,
                        "upload_date": datetime.now().isoformat(),
                        "source": "upload",
                    })
                    # Immediately index into LanceDB vector store
                    try:
                        embed_start = time.time()
                        lancedb_chat = init_lancedb_chat()
                        with st.spinner("Indexing document into LanceDB..."):
                            lancedb_chat.add_documents(st.session_state.lancedb_documents[-1:])
                        embed_duration = (time.time() - embed_start) * 1000
                        
                        # Track LanceDB storage
                        embedding_ids = [f"embedding_{i}" for i in range(extraction_result.get('sections_count', 0))]
                        lineage.track_lancedb_storage(embedding_ids, f"doc_{doc_id_parsed}", embed_duration)
                        
                        st.session_state.lancedb_indexed = True
                    except Exception as e:
                        st.session_state.lancedb_indexed = False
                        st.warning(f"LanceDB indexing deferred: {e}")
                
                # Show extraction summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Pages", extraction_result.get('pages_count', 0))
                with col2:
                    st.metric("📑 Sections", extraction_result.get('sections_count', 0))
                with col3:
                    st.metric("🔍 Entities", extraction_result.get('entities_found', 0))
                
            st.success(f"✅ {uploaded_file.name} parsed and stored in LanceDB!")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        st.rerun()
        
        st.markdown("---")

    # LanceDB document picker
    lance_docs = init_pdf_parser().list_lancedb_documents()
    with st.expander("🗄️ Documents in LanceDB", expanded=False):
        if not lance_docs:
            st.info("No documents found in LanceDB.")
            if st.button("🔄 Refresh LanceDB list", key="refresh_lancedb_docs"):
                st.rerun()
        else:
            options = {f"{d.get('document_name', 'doc')} ({d.get('document_id','')[:8]})": d for d in lance_docs}
            selections = st.multiselect(
                "Select cached document(s) to chat",
                options=list(options.keys()),
                key="lancedb_doc_select_multi"
            ) if options else []

            if selections and st.button("Load selected for chat", key="load_lancedb_docs"):
                loaded = 0
                docs_to_index = []
                message_sources = st.session_state.get("message_source_ids", set()) or set()
                if "lancedb_documents" not in st.session_state:
                    st.session_state.lancedb_documents = []
                for selection in selections:
                    chosen_meta = options[selection]
                    source = hydrate_source_from_lancedb(chosen_meta)
                    if not source:
                        st.error(f"Could not load sections from LanceDB for {chosen_meta.get('document_name')}")
                        continue

                    st.session_state.sources[source.id] = source
                    message_sources.add(source.id)

                    doc_id = chosen_meta.get("document_id")
                    # Ensure the doc is tracked for LanceDB chat mode
                    if not any(doc.get("document_id") == doc_id for doc in st.session_state.lancedb_documents):
                        doc_record = {
                            "name": chosen_meta.get("document_name", source.name),
                            "document_id": doc_id,
                            "content": source.content,
                            "file_type": chosen_meta.get("file_type", "pdf"),
                            "file_size": chosen_meta.get("file_size", 0),
                            "upload_date": chosen_meta.get("upload_date"),
                            "source": "lancedb_cache",
                        }
                        st.session_state.lancedb_documents.append(doc_record)
                        docs_to_index.append(doc_record)
                    else:
                        # Reuse existing stored record for re-indexing if needed
                        existing = next(doc for doc in st.session_state.lancedb_documents if doc.get("document_id") == doc_id)
                        docs_to_index.append(existing)

                    # Compute section count with fallback
                    section_count = len(load_sections_from_lancedb(doc_id))
                    if section_count == 0:
                        try:
                            store = get_lancedb_store()
                            section_count = len(store.load_sections(doc_id))
                        except Exception:
                            section_count = 0

                    st.session_state.parsed_documents[source.id] = {
                        'document_name': chosen_meta.get('document_name', source.name),
                        'document_id': doc_id,
                        'sections_count': section_count,
                        'extraction_result': {
                            'document_id': doc_id,
                            'document_name': chosen_meta.get('document_name'),
                            'pages': chosen_meta.get('pages', 0)
                        }
                    }
                    loaded += 1

                # Documents are already indexed in LanceDB (from initial upload)
                # Just mark as indexed for chat mode
                st.session_state.lancedb_indexed = True

                st.session_state.message_source_ids = message_sources
                if loaded:
                    st.success(f"Loaded {loaded} document(s) from LanceDB for chat")
                    st.rerun()
 
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
                            parser.load_document_from_lancedb(doc_id)
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
                            pdf_parser.load_document_from_lancedb(doc_id)
                            sections = load_sections_from_lancedb(doc_id)
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
                        pdf_parser.load_document_from_lancedb(doc_id)
                        entities = pdf_parser.search_parser.search_engine.search_entities(
                            doc_id,
                            None if entity_type_filter == "All" else entity_type_filter
                        )
                        sections = load_sections_from_lancedb(doc_id)
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
            st.rerun()
if not st.session_state.sources:
    st.info("Add your documents in the sidebar to start chatting", icon="👈")
elif st.session_state.sources:
    for message in st.session_state.messages:
        avatar_icon = "🤖" if message["role"] == "assistant" else "🐧"
        with st.chat_message(message["role"], avatar=avatar_icon):
            thinking_content = message.get("thinking")
            if thinking_content:
                with st.expander("CAG's thoughts", expanded=False):
                    st.markdown(thinking_content)
 
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_skill_list(message.get("skills"), label="Skills used", expanded=False)
                
                # Voice Output for Assistant Messages
                if st.session_state.voice_enabled and st.session_state.voice_output_enabled:
                    col1, col2 = st.columns([0.3, 0.7])
                    with col1:
                        if st.button("🔊 Speak", key=f"voice_speak_{id(message)}", use_container_width=True):
                            with st.spinner("Synthesizing speech..."):
                                try:
                                    voice_config = VoiceConfig(
                                        tts_rate=st.session_state.get("voice_tts_rate", 150),
                                        tts_volume=st.session_state.get("voice_tts_volume", 1.0)
                                    )
                                    voice_processor = get_voice_processor(voice_config)
                                    audio_bytes = voice_processor.synthesize_answer(message["content"])
                                    
                                    if audio_bytes:
                                        st.session_state[f"answer_audio_{id(message)}"] = audio_bytes
                                        st.success("✅ Audio ready")
                                    else:
                                        st.error("Speech synthesis failed")
                                except Exception as e:
                                    st.error(f"TTS error: {str(e)}")
                    
                    with col2:
                        audio_key = f"answer_audio_{id(message)}"
                        if audio_key in st.session_state:
                            st.audio(st.session_state[audio_key], format="audio/wav")
            # Link section references in assistant replies to full sections from LanceDB
            if message["role"] == "assistant" and st.session_state.parsed_documents:
                # Collect sections for the current message's documents
                all_sections = []
                for source_id, parsed_info in st.session_state.parsed_documents.items():
                    doc_id = parsed_info.get("extraction_result", {}).get("document_id")
                    if not doc_id:
                        continue
                    all_sections.extend(load_sections_from_lancedb(doc_id))
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
    
    # Voice Input Recording
    if st.session_state.voice_enabled and st.session_state.voice_input_enabled and chat_enabled:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🎙️ Record Question", key="voice_record_btn"):
                with st.spinner("Recording..."):
                    try:
                        voice_processor = get_voice_processor(
                            VoiceConfig(
                                record_duration_max=st.session_state.get("voice_record_duration", 10)
                            )
                        )
                        audio_bytes, sample_rate = voice_processor.record_audio(
                            st.session_state.get("voice_record_duration", 10)
                        )
                        
                        if audio_bytes:
                            st.session_state.last_audio = audio_bytes
                            st.success("✅ Recording saved. Transcribing...")
                    except Exception as e:
                        st.error(f"Recording failed: {str(e)}")
        
        with col2:
            if st.button("📝 Transcribe", key="voice_transcribe_btn", disabled="last_audio" not in st.session_state):
                if "last_audio" in st.session_state:
                    with st.spinner("Transcribing..."):
                        try:
                            voice_processor = get_voice_processor()
                            question = voice_processor.transcribe_and_extract_question(
                                st.session_state.last_audio
                            )
                            
                            if question:
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": question,
                                    "thinking": None
                                })
                                st.session_state.message_source_ids = set(st.session_state.sources.keys())
                                st.success(f"✓ Transcribed: {question}")
                                st.rerun()
                            else:
                                st.error("Transcription failed")
                        except Exception as e:
                            st.error(f"Transcription error: {str(e)}")
        
        with col3:
            if "last_audio" in st.session_state:
                st.audio(st.session_state.last_audio, format="audio/wav")
    
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
            query = st.session_state.messages[-1]["content"]
            history = st.session_state.messages[:-1]
            detected_skills = infer_skills_from_question(query)
            
            # Check if LanceDB mode is enabled and documents are indexed
            use_lancedb = (
                st.session_state.lancedb_chat_mode and 
                st.session_state.lancedb_indexed and 
                len(st.session_state.lancedb_documents) > 0
            )
            
            if use_lancedb:
                # Use LanceDB for multi-document chat
                try:
                    lancedb_chat = init_lancedb_chat()
                    search_mode_display = st.session_state.lancedb_search_mode.title()
                    
                    # Build filter info for display
                    filter_info = ""
                    if st.session_state.get("lancedb_filter_enabled"):
                        if st.session_state.get("lancedb_pre_filter") or st.session_state.get("lancedb_post_filter"):
                            filter_info = " with filters"
                    
                    # Build advanced FTS config if enabled
                    advanced_search_config = None
                    if st.session_state.get("lancedb_fts_enabled") and st.session_state.lancedb_search_mode == "hybrid":
                        from lancedb_chat import AdvancedSearchConfig
                        
                        advanced_search_config = AdvancedSearchConfig(
                            query_type=st.session_state.get("lancedb_fts_query_type", "match"),
                            fields=st.session_state.get("lancedb_fts_fields"),
                            field_boosts=st.session_state.get("lancedb_field_boosts"),
                            boost=st.session_state.get("lancedb_overall_boost", 1.0),
                            phrase_slop=st.session_state.get("lancedb_phrase_slop", 0),
                            fuzzy_distance=st.session_state.get("lancedb_fuzzy_distance", 2),
                            boolean_operator=st.session_state.get("lancedb_boolean_operator", "AND"),
                            use_multivector=st.session_state.get("lancedb_multivector_enabled", False),
                            multivector_queries=st.session_state.get("lancedb_query_variations"),
                        )
                        fts_type = advanced_search_config.query_type
                        cache_status_placeholder.info(f"🗄️ Using LanceDB {search_mode_display} search ({fts_type}){filter_info}")
                    else:
                        cache_status_placeholder.info(f"🗄️ Using LanceDB {search_mode_display} search{filter_info}")
                    
                    full_response_content = ""
                    
                    # Query LanceDB with filters and advanced FTS
                    filter_config = FilterConfig(
                        pre_filter=st.session_state.get("lancedb_pre_filter"),
                        post_filter=st.session_state.get("lancedb_post_filter"),
                    )
                    response = lancedb_chat.query(
                        query, 
                        filter_config=filter_config,
                        advanced_search=advanced_search_config,
                    )
                    full_response_content = response.answer
                    
                    # Track retrieval in lineage
                    retrieve_end = time.time()
                    if response.source_nodes:
                        lineage.track_retrieval(
                            query=query,
                            retrieved_sections=list(response.source_nodes)[:5],  # Top 5 sources
                            duration_ms=(retrieve_end - start_time) * 1000,
                            cache_hit=getattr(response, 'from_cache', False)
                        )
                        query_id = f"query_{retrieve_end}"
                    
                    # Track LLM response
                    lineage.track_llm_response(
                        query_id=query_id if response.source_nodes else "",
                        response=full_response_content,
                        duration_ms=100,  # LanceDB response time typically < 100ms
                        model=current_model_config.model_name
                    )
                    
                    # Display response
                    message_placeholder.markdown(full_response_content)
                    render_skill_list(detected_skills, label="Skills used", expanded=False)
                    
                    # Show source documents
                    if response.source_nodes:
                        with st.expander("📚 Sources", expanded=False):
                            for source_name in response.source_nodes:
                                st.caption(f"📄 {source_name}")

                    # Show chunk references (hybrid skill output)
                    if getattr(response, "contexts", None):
                        with st.expander("🔎 Retrieved Chunks", expanded=False):
                            for idx, ctx in enumerate(response.contexts[:10]):
                                st.markdown(f"**Chunk {idx+1}:** {ctx.get('filename', 'Unknown')}")
                                st.caption(ctx.get("text", "")[:400] + ("..." if len(ctx.get("text", "")) > 400 else ""))
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response_content,
                        "thinking": None,
                        "skills": detected_skills,
                    })
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"LanceDB query failed: {str(e)}")
                    st.info("Falling back to regular chat mode")
                    use_lancedb = False
            
            # Check if Agentic RAG mode is enabled
            use_agentic_rag = (
                st.session_state.get("agentic_rag_mode", False) and
                len(st.session_state.parsed_documents) > 0 and
                not use_lancedb  # Don't use both modes simultaneously
            )
            
            if use_agentic_rag:
                # Use Agentic RAG for multi-step reasoning
                try:
                    cache_status_placeholder.info("🤖 Using Agentic RAG with multi-step reasoning...")
                    
                    # Get document IDs
                    doc_ids = list(set(
                        parsed_info['document_id']
                        for parsed_info in st.session_state.parsed_documents.values()
                    ))
                    
                    # Initialize agentic RAG
                    agentic_rag = init_agentic_rag(current_model_config)
                    
                    # Get configuration
                    max_context = st.session_state.get("agentic_max_context", 16000)
                    
                    # Execute agentic query
                    with st.spinner("Agent is reasoning about your query..."):
                        answer = agentic_rag.answer_query(
                            query=query,
                            document_ids=doc_ids,
                            max_context_length=max_context
                        )
                    
                    # Display the answer
                    message_placeholder.markdown(answer.answer)
                    render_skill_list(detected_skills, label="Skills used", expanded=False)
                    
                    # Show reasoning steps if enabled
                    if st.session_state.get("agentic_show_reasoning", True):
                        with st.expander("🧠 Agent Reasoning Process", expanded=False):
                            st.markdown("**Query Plan:**")
                            st.json({
                                "Intent": answer.plan.intent,
                                "Strategy": answer.plan.strategy.value,
                                "Keywords": answer.plan.keywords,
                                "Expected Sections": answer.plan.expected_section_types
                            })
                            
                            st.markdown("**Reasoning Steps:**")
                            for i, step in enumerate(answer.reasoning_steps, 1):
                                st.caption(f"{i}. {step}")
                            
                            if answer.plan.reasoning:
                                st.markdown("**Strategy Reasoning:**")
                                st.info(answer.plan.reasoning)
                    
                    # Show thinking if available
                    if answer.thinking:
                        with st.expander("💭 Agent Thinking", expanded=False):
                            st.markdown(answer.thinking)
                    
                    # Show sources
                    if answer.sources:
                        with st.expander(f"📚 Sources ({len(answer.sources)})", expanded=False):
                            for ctx in answer.sources:
                                st.markdown(f"**{ctx.section_title or ctx.source_id}**")
                                st.caption(f"Relevance: {ctx.relevance_score:.2f}")
                                if ctx.reasoning:
                                    st.info(f"Why: {ctx.reasoning}")
                                st.caption(ctx.content[:300] + ("..." if len(ctx.content) > 300 else ""))
                                st.markdown("---")
                    
                    # Show MCP tool usage
                    if hasattr(answer, 'tool_usage_history') and answer.tool_usage_history:
                        with st.expander(f"🔧 MCP Tools Used ({len(answer.tool_usage_history)})", expanded=False):
                            for tool_result in answer.tool_usage_history:
                                st.markdown(f"**🛠️ {tool_result.tool_name}**")
                                st.caption(f"Executed at: {tool_result.timestamp}")
                                
                                if tool_result.success:
                                    st.success(f"✅ Completed in {tool_result.execution_time:.2f}s")
                                    
                                    # Display result data based on tool type
                                    if tool_result.tool_name == "web_search_augment" and tool_result.result:
                                        st.markdown("**External Sources:**")
                                        for source in tool_result.result.get("sources", []):
                                            st.markdown(f"- [{source.get('title', 'Source')}]({source.get('url', '#')})")
                                            st.caption(source.get("snippet", "")[:200])
                                    
                                    elif tool_result.tool_name == "extract_entities_from_context" and tool_result.result:
                                        entities = tool_result.result.get("entities", {})
                                        if entities:
                                            st.markdown("**Extracted Entities:**")
                                            for entity_type, entity_list in entities.items():
                                                if entity_list:
                                                    st.markdown(f"- **{entity_type}**: {', '.join(entity_list[:10])}")
                                    
                                    elif tool_result.tool_name == "rank_sections_by_importance" and tool_result.result:
                                        rankings = tool_result.result.get("rankings", [])
                                        if rankings:
                                            st.markdown("**Top Ranked Sections:**")
                                            for rank in rankings[:5]:
                                                st.markdown(f"- {rank['title']} (Score: {rank['score']:.2f})")
                                    
                                    elif tool_result.tool_name == "find_cross_document_relationships" and tool_result.result:
                                        relationships = tool_result.result.get("relationships", [])
                                        if relationships:
                                            st.markdown(f"**Found {len(relationships)} Cross-References:**")
                                            for rel in relationships[:3]:
                                                st.markdown(f"- {rel['type']}: {rel.get('description', 'N/A')}")
                                    
                                    elif tool_result.tool_name == "verify_fact_with_web" and tool_result.result:
                                        st.markdown(f"**Verification Status:** {tool_result.result.get('verification_status', 'Unknown')}")
                                        if tool_result.result.get("evidence"):
                                            st.info(f"Evidence: {tool_result.result['evidence'][:200]}...")
                                    
                                    elif tool_result.tool_name == "suggest_follow_up_questions" and tool_result.result:
                                        questions = tool_result.result.get("questions", [])
                                        if questions:
                                            st.markdown("**Suggested Follow-ups:**")
                                            for q in questions:
                                                st.markdown(f"- {q}")
                                    
                                    # Show metadata if available
                                    if tool_result.metadata:
                                        with st.expander("ℹ️ Tool Metadata", expanded=False):
                                            st.json(tool_result.metadata)
                                else:
                                    st.error(f"❌ Failed: {tool_result.error}")
                                
                                st.markdown("---")
                    
                    # Show confidence and validation
                    col1, col2 = st.columns(2)
                    with col1:
                        confidence_color = "🟢" if answer.confidence > 0.7 else "🟡" if answer.confidence > 0.5 else "🔴"
                        st.metric("Confidence", f"{confidence_color} {answer.confidence:.1%}")
                    with col2:
                        st.metric("Strategy", answer.plan.strategy.value.title())
                    
                    if answer.validation_notes:
                        with st.expander("✅ Quality Validation", expanded=False):
                            st.markdown(answer.validation_notes)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer.answer,
                        "thinking": answer.thinking,
                        "skills": detected_skills,
                    })
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Agentic RAG failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("Falling back to regular chat mode")
                    use_agentic_rag = False
            
            if not use_lancedb and not use_agentic_rag:
                # Use regular chat mode
                llm = create_chatbot(current_model_config)
            
            # Get document IDs for this query
            doc_ids = set()
            for source_id, parsed_info in st.session_state.parsed_documents.items():
                doc_id = parsed_info.get("extraction_result", {}).get("document_id")
                if doc_id:
                    doc_ids.add(doc_id)
            
            # Check Q&A cache
            qa_cache = get_qa_cache()
            question_library = get_question_library()
            cached_result = qa_cache.get_cached_answer(question=query, doc_ids=doc_ids) if doc_ids else None
            
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
                render_skill_list(detected_skills, label="Skills used", expanded=False)
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
                render_skill_list(detected_skills, label="Skills used", expanded=False)
                
                # Cache the response
                if doc_ids:
                    qa_cache.cache_answer(
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
                    "skills": detected_skills,
                }
            )
            st.rerun()