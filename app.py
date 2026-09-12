import os
import sys
import json
import time
import pandas as pd
import streamlit as st

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.parser import parse_all_documents, parse_custom_file
from src.retriever import Retriever
from src.verifier import Verifier
from eval import run_evaluation

st.set_page_config(
    page_title="The Rulebook That Argues With Itself",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS & Modern Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
        margin-top: 0.25rem;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    .badge-answers {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        font-weight: 700;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
    }
    
    .badge-contradiction {
        display: inline-block;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff;
        font-weight: 700;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3);
    }
    
    .badge-unanswered {
        display: inline-block;
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: #ffffff;
        font-weight: 700;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(107, 114, 128, 0.3);
    }
    
    .card-conflict {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #ef4444;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .card-conflict-title {
        font-weight: 700;
        color: #991b1b;
        margin-bottom: 0.5rem;
        font-size: 1.05rem;
    }

    .resolution-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #10b981;
        border-radius: 10px;
        padding: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .resolution-title {
        font-weight: 800;
        color: #166534;
        font-size: 1.15rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    chunks = parse_all_documents()
    retriever = Retriever()
    retriever.build_index(chunks)
    verifier = Verifier()
    return retriever, verifier, len(chunks)

try:
    retriever, verifier, total_chunks_count = get_pipeline()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# Sidebar Info & Controls
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    st.success("🟢 **Groq Verifier**: Ready", icon="⚡")
    st.info("📦 **Vector Storage**: ChromaDB Active", icon="📚")
    st.metric(label="Total Policy Chunks Indexed", value=total_chunks_count)
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Benchmark Presets")
    example_queries = [
        "-- Select a benchmark preset --",
        "Who controls the official academic record when an informal advising comment differs from registrar records?",
        "Can a student with 68% attendance sit a final exam if the instructor says strong continuous evaluation performance justifies it?",
        "Who has final authority to waive a prerequisite course: the Dean of Academic Affairs or the Department Head for a major?",
        "For a medical absence, does the student need hospital documentation within 48 hours of missed work or Campus Health Center validation within seven business days after returning?",
        "May a student miss a laboratory practical to attend a sibling's wedding and still receive attendance credit?"
    ]
    selected_preset = st.selectbox("Choose a test case:", example_queries)

    st.markdown("---")
    st.caption("⚡ Powered by Groq API (`llama-3.3-70b-versatile`) & SentenceTransformers.")

# Header Title
st.markdown('<div class="main-title">The Rulebook That Argues With Itself</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Policy Verification, Contradiction Detection & Automatic Rule Reconciliation Engine</div>', unsafe_allow_html=True)

# Main Application Tabs
tab_verify, tab_dashboard, tab_kb = st.tabs([
    "⚖️ Policy Verifier & Resolver",
    "📊 Benchmark Analytics",
    "📂 Document Manager"
])

# ==========================================
# TAB 1: POLICY VERIFIER & RESOLVER
# ==========================================
with tab_verify:
    default_text = "" if selected_preset == "-- Select a benchmark preset --" else selected_preset
    
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        query_input = st.text_input(
            "Enter policy query or regulatory question:",
            value=default_text,
            placeholder="e.g., What is the minimum attendance requirement for final examinations?"
        )
    with col_btn:
        st.write("") # Spacer
        st.write("")
        verify_clicked = st.button("Verify Policy Query", type="primary", use_container_width=True)

    if verify_clicked:
        if not query_input.strip():
            st.warning("Please enter a query to verify.")
        else:
            with st.spinner("Retrieving candidate passages & verifying policy alignment with Groq..."):
                start_t = time.time()
                retrieved = retriever.retrieve(query_input, top_k=5)
                result = verifier.verify(query_input, retrieved)
                proc_time = round(time.time() - start_t, 2)
                
                # Store in session state for downstream actions (e.g. resolution generation / export)
                st.session_state["current_result"] = result
                st.session_state["current_query"] = query_input
                st.session_state["current_retrieved"] = retrieved
                st.session_state["proc_time"] = proc_time
                # Reset resolution state
                if "current_resolution" in st.session_state:
                    del st.session_state["current_resolution"]

    if "current_result" in st.session_state:
        result = st.session_state["current_result"]
        query_input = st.session_state["current_query"]
        retrieved = st.session_state["current_retrieved"]
        proc_time = st.session_state.get("proc_time", 0.0)
        
        state = result["state"]
        answer = result["answer"]
        citations = result["citations"]

        st.markdown("---")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("### Verification Analysis")
            if state == "ANSWERS":
                st.markdown('<span class="badge-answers">STATE: ANSWERS</span>', unsafe_allow_html=True)
            elif state == "CONTRADICTION":
                st.markdown('<span class="badge-contradiction">STATE: CONTRADICTION DETECTED ⚔️</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-unanswered">STATE: UNANSWERED</span>', unsafe_allow_html=True)
        with c2:
            st.caption(f"⏱️ Response Latency: `{proc_time}s`")
            
        st.info(f"**Policy Narrative:**\n\n{answer}")

        # Contradiction Side-by-Side Cards
        if state == "CONTRADICTION" and len(citations) >= 2:
            st.markdown("### ⚔️ Side-by-Side Contradictory Provisions")
            col_a, col_b = st.columns(2)
            with col_a:
                cit1 = citations[0]
                st.markdown(f"""
                <div class="card-conflict">
                    <div class="card-conflict-title">Provision A: {cit1.get('file', '')} ({cit1.get('section', '')})</div>
                    <p>"{cit1.get('excerpt', '')}"</p>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                cit2 = citations[1]
                st.markdown(f"""
                <div class="card-conflict">
                    <div class="card-conflict-title">Provision B: {cit2.get('file', '')} ({cit2.get('section', '')})</div>
                    <p>"{cit2.get('excerpt', '')}"</p>
                </div>
                """, unsafe_allow_html=True)

        # Policy Resolution Engine Button & Interface
        if state == "CONTRADICTION":
            st.markdown("### 🛠️ Policy Resolution & Reconciliation Engine")
            if st.button("Generate Policy Resolution Proposal 💡", type="secondary"):
                with st.spinner("Synthesizing unified policy redraft and hierarchy of authority with Groq..."):
                    resolution = verifier.resolve_contradiction(query_input, citations, answer)
                    st.session_state["current_resolution"] = resolution

            if "current_resolution" in st.session_state:
                res = st.session_state["current_resolution"]
                st.markdown(f"""
                <div class="resolution-card">
                    <div class="resolution-title">✅ Reconciled Policy Proposal</div>
                    <p><b>Unified Rule:</b> {res.get('reconciled_rule')}</p>
                    <p><b>Hierarchy of Authority:</b> {res.get('recommended_hierarchy')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 📝 Proposed Verbatim Policy Redraft:")
                st.code(res.get("proposed_redraft", ""), language="markdown")

        # Citations Section
        if citations:
            with st.expander("📚 View All Citations & Passages", expanded=(state != "CONTRADICTION")):
                for idx, cit in enumerate(citations, 1):
                    st.markdown(f"**Citation #{idx}** - `{cit.get('file', '')}` | Section: `{cit.get('section', '')}`")
                    st.caption(f'"{cit.get("excerpt", "")}"')

        with st.expander("🔍 View Raw Retrieved Chunks (Top 5)"):
            for idx, chunk in enumerate(retrieved, 1):
                st.markdown(f"**Chunk #{idx}** | `{chunk['file']}` | Section: `{chunk['section']}` | Distance Score: `{chunk['score']:.4f}`")
                st.code(chunk['content'], language="markdown")

        # Export Report Actions
        st.markdown("---")
        st.markdown("### 📥 Export Audit Report")
        
        # Prepare Markdown export string
        md_export = f"""# Policy Verification Audit Report
**Query:** {query_input}
**State:** {state}
**Generated At:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Answer Narrative
{answer}

## Citations
"""
        for idx, c in enumerate(citations, 1):
            md_export += f"- Citation #{idx}: [{c.get('file')}] Section: {c.get('section')}\n  Excerpt: \"{c.get('excerpt')}\"\n"
            
        if "current_resolution" in st.session_state:
            res = st.session_state["current_resolution"]
            md_export += f"\n## Policy Reconciliation Proposal\n"
            md_export += f"- Reconciled Rule: {res.get('reconciled_rule')}\n"
            md_export += f"- Recommended Hierarchy: {res.get('recommended_hierarchy')}\n"
            md_export += f"- Proposed Redraft:\n```markdown\n{res.get('proposed_redraft')}\n```\n"

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                label="📄 Download Audit Report (.MD)",
                data=md_export,
                file_name="policy_verification_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        with exp_col2:
            st.download_button(
                label="📊 Download Raw Audit Data (.JSON)",
                data=json.dumps(result, indent=2),
                file_name="policy_verification_result.json",
                mime="application/json",
                use_container_width=True
            )

# ==========================================
# TAB 2: BENCHMARK ANALYTICS DASHBOARD
# ==========================================
with tab_dashboard:
    st.markdown("### 📊 Benchmark Performance & Accuracy Dashboard")
    
    # Load evaluation results if available
    eval_file = "eval_results.json"
    eval_data = None
    if os.path.exists(eval_file):
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            eval_data = None

    col_btn_eval, col_info_eval = st.columns([1, 3])
    with col_btn_eval:
        if st.button("🚀 Run Live Evaluation Benchmark", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Running evaluation benchmark on 38 queries...")
            with st.spinner("Executing pipeline evaluation..."):
                run_evaluation(dataset_path="eval_dataset.json", output_path="eval_results.json")
                progress_bar.progress(100, text="Benchmark completed!")
                st.success("Evaluation complete! Reloading metrics...")
                time.sleep(1)
                st.rerun()

    if eval_data:
        summary = eval_data.get("summary", {})
        results = eval_data.get("results", [])
        
        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Accuracy</div>
                <div class="metric-value">{summary.get('overall_accuracy_pct', 0)}%</div>
                <div class="metric-sub">{summary.get('total_queries', 0)} Total Queries</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Contradiction Detection</div>
                <div class="metric-value">{summary.get('contradiction_accuracy_pct', 0)}%</div>
                <div class="metric-sub">{summary.get('contradiction_accuracy', '0/0')} Queries Correct</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Unanswerable Rejection</div>
                <div class="metric-value">{summary.get('unanswerable_rejection_rate_pct', 0)}%</div>
                <div class="metric-sub">{summary.get('unanswerable_rejection_rate', '0/0')} Queries Correct</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Direct Answers Accuracy</div>
                <div class="metric-value">{summary.get('answers_accuracy_pct', 0)}%</div>
                <div class="metric-sub">{summary.get('answers_accuracy', '0/0')} Queries Correct</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Charts Section
        chart_col1, chart_col2 = st.columns(2)
        df_res = pd.DataFrame(results)
        
        with chart_col1:
            st.markdown("#### State Distribution Breakdown")
            if not df_res.empty:
                state_counts = df_res['predicted_state'].value_counts()
                st.bar_chart(state_counts)
                
        with chart_col2:
            st.markdown("#### Accuracy by Expected State")
            if not df_res.empty:
                acc_by_state = df_res.groupby('expected_state')['is_correct'].mean() * 100
                st.bar_chart(acc_by_state)

        st.markdown("---")
        st.markdown("#### 📋 Detailed Benchmark Query Breakdown")
        
        # Filter options
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_filter_state = st.selectbox(
                "Filter by Expected State:",
                ["ALL", "ANSWERS", "CONTRADICTION", "UNANSWERED"]
            )
        with filter_col2:
            selected_filter_status = st.selectbox(
                "Filter by Status:",
                ["ALL", "CORRECT ONLY", "MISMATCH ONLY"]
            )

        df_display = df_res.copy()
        if selected_filter_state != "ALL":
            df_display = df_display[df_display["expected_state"] == selected_filter_state]
        if selected_filter_status == "CORRECT ONLY":
            df_display = df_display[df_display["is_correct"] == True]
        elif selected_filter_status == "MISMATCH ONLY":
            df_display = df_display[df_display["is_correct"] == False]

        show_cols = ["id", "query", "expected_state", "predicted_state", "is_correct"]
        st.dataframe(
            df_display[show_cols],
            use_container_width=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "query": st.column_config.TextColumn("Query", width="large"),
                "expected_state": st.column_config.TextColumn("Expected", width="medium"),
                "predicted_state": st.column_config.TextColumn("Predicted", width="medium"),
                "is_correct": st.column_config.CheckboxColumn("Correct?", width="small")
            }
        )

    else:
        st.info("No benchmark results found. Click 'Run Live Evaluation Benchmark' above to execute evaluation.")

# ==========================================
# TAB 3: DOCUMENT MANAGER & KNOWLEDGE BASE
# ==========================================
with tab_kb:
    st.markdown("### 📂 Document Upload & Vector Knowledge Base Manager")
    st.caption("Upload new policy manuals, PDFs, CSV fee schedules, or Markdown documents to dynamically expand and re-index the ChromaDB vector store.")

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        uploaded_files = st.file_uploader(
            "Upload Policy Documents (.md, .pdf, .csv, .txt):",
            type=["md", "pdf", "csv", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("📥 Save & Re-index Corpus", type="primary"):
                saved_files = []
                data_dir = "data"
                os.makedirs(data_dir, exist_ok=True)
                
                for uf in uploaded_files:
                    target_path = os.path.join(data_dir, uf.name)
                    with open(target_path, "wb") as f:
                        f.write(uf.getbuffer())
                    saved_files.append(uf.name)
                
                st.success(f"Successfully saved {len(saved_files)} file(s): {', '.join(saved_files)}")
                
                with st.spinner("Parsing documents and rebuilding ChromaDB vector index..."):
                    all_chunks = parse_all_documents()
                    retriever.build_index(all_chunks, force_rebuild=True)
                    st.cache_resource.clear()
                    st.success(f"Successfully re-indexed {len(all_chunks)} chunks into ChromaDB!", icon="🎉")
                    time.sleep(1)
                    st.rerun()

    with up_col2:
        st.markdown("### 📊 Vector Index Metadata")
        data_dir = "data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
            st.metric("Total Corpus Files", len(files))
            for f in files:
                fpath = os.path.join(data_dir, f)
                fsize = round(os.path.getsize(fpath) / 1024, 2)
                st.text(f"• {f} ({fsize} KB)")

    st.markdown("---")
    st.markdown("### 🔍 Current Parsed Chunks Preview")
    all_chunks = parse_all_documents()
    df_chunks = pd.DataFrame(all_chunks)
    if not df_chunks.empty:
        st.dataframe(
            df_chunks[["chunk_id", "file", "section", "page"]],
            use_container_width=True,
            height=300
        )
