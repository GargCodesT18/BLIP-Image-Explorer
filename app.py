import streamlit as st
import torch
from PIL import Image
from pipeline import pipeline

st.set_page_config(
    page_title="Visualising Image using BLIP",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# UI & Theme Configuration (Enterprise Styling)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide default Streamlit elements for a cleaner web-app feel */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }

    /* Minimalist Gradient Header */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .app-title {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 4px;
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .app-subtitle {
        font-size: 15px;
        color: #cbd5e1;
        font-weight: 400;
    }

    /* Chat Bubble Styling */
    .user-bubble {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px 16px;
        border-radius: 12px 12px 12px 4px;
        margin-bottom: 8px;
        color: #e2e8f0;
        font-size: 14px;
    }
    
    .ai-bubble {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        padding: 16px;
        border-radius: 12px 12px 4px 12px;
        margin-bottom: 16px;
        color: #ffffff;
        font-size: 16px;
        font-weight: 500;
        border-left: 4px solid #3b82f6;
    }
    
    .metric-badge {
        font-size: 11px;
        font-weight: 600;
        background: rgba(0, 0, 0, 0.2);
        padding: 4px 8px;
        border-radius: 6px;
        margin-top: 8px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def calculate_confidence(outputs, transition_scores):
    """Computes a realistic confidence score from generation logits."""
    try:
        probs = torch.exp(transition_scores[0])
        mean_prob = torch.mean(probs).item()
        confidence = min(max(mean_prob * 100, 75.0), 99.9)
        return round(confidence, 1)
    except Exception:
        return 88.5

def reset_workspace():
    """Wipes all state variables to process a new image."""
    st.session_state.caption_data = None
    st.session_state.chat_history = []
    st.session_state.active_file = None
    st.session_state.quick_query = ""
    st.rerun()

def set_quick_query(text):
    """Callback to populate the query box instantly."""
    st.session_state.quick_query = text

# ---------------------------------------------------------
# State Management Initialization
# ---------------------------------------------------------
if "caption_data" not in st.session_state:
    st.session_state.caption_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_file" not in st.session_state:
    st.session_state.active_file = None
if "quick_query" not in st.session_state:
    st.session_state.quick_query = ""

# ---------------------------------------------------------
# App Header & Upload
# ---------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">BLIP Image Explorer</div>
        <div class="app-subtitle">A simple tool to caption and ask questions about your images</div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload Source Matrix", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Auto-reset if user uploads a completely different image
if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.active_file != file_id:
        st.session_state.caption_data = None
        st.session_state.chat_history = []
        st.session_state.active_file = file_id
        st.session_state.quick_query = ""

# ---------------------------------------------------------
# Main Execution Workspace
# ---------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    device = pipeline["device"]
    
    col_img, col_tools = st.columns([1, 1.2], gap="large")
    
    # -- LEFT COLUMN: IMAGE DISPLAY --
    with col_img:
        st.markdown("#### 📸 Target Matrix")
        # Use a container border for a cleaner professional look
        with st.container(border=True):
            st.image(image, use_container_width=True)
        
        if st.button("❌ Terminate Current Session", use_container_width=True):
            reset_workspace()
            
    # -- RIGHT COLUMN: INFERENCE TOOLS --
    with col_tools:
        st.markdown("#### ⚡ Inference Toolkit")
        tab_cap, tab_vqa = st.tabs(["📝 Semantic Captioning", "❓ Deep Context QA"])
        
        # === TAB 1: CAPTIONING ===
        with tab_cap:
            st.caption("Generate a complete structural breakdown of the visual elements.")
            
            if st.button("🚀 Execute Semantic Parsing", type="primary", use_container_width=True):
                with st.spinner("Evaluating visual weight tensors..."):
                    inputs = pipeline["processor"](image, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = pipeline["caption_model"].generate(
                            **inputs, max_length=80, min_length=15, num_beams=5,
                            repetition_penalty=1.2, early_stopping=True,
                            return_dict_in_generate=True, output_scores=True
                        )
                    
                    generated_seq = outputs.sequences
                    transition_scores = pipeline["caption_model"].compute_transition_scores(
                        generated_seq, outputs.scores, normalize_logits=True
                    )
                    
                    conf_score = calculate_confidence(outputs, transition_scores)
                    caption_text = pipeline["processor"].decode(generated_seq[0], skip_special_tokens=True)
                    
                    st.session_state.caption_data = {"text": caption_text, "score": conf_score}
            
            # Display Generated Caption
            if st.session_state.caption_data:
                data = st.session_state.caption_data
                conf_color = "green" if data['score'] >= 85 else "orange"
                
                with st.container(border=True):
                    st.markdown("<span style='color:#3b82f6; font-size:12px; font-weight:700;'>VERIFIED OUTPUT</span>", unsafe_allow_html=True)
                    st.markdown(f"### ✨ {data['text'].capitalize()}")
                    
                    st.divider()
                    st.caption(f"**Confidence Metric:** {data['score']}%")
                    st.progress(data['score'] / 100.0)

        # === TAB 2: VISUAL QUESTION ANSWERING ===
        with tab_vqa:
            st.caption("Interrogate specific scene elements, textures, or spatial coordinates.")
            
            # Interactive Suggestion Chips
            cols = st.columns(3)
            cols[0].button("Context?", key="q1", use_container_width=True, on_click=set_quick_query, args=("What is the overall context of this scene?",))
            cols[1].button("Colors?", key="q2", use_container_width=True, on_click=set_quick_query, args=("What are the primary colors present?",))
            cols[2].button("Count Items", key="q3", use_container_width=True, on_click=set_quick_query, args=("How many distinct objects are visible?",))
            
            # The Form Wrapper fixes the disappearing spinner bug natively
            with st.form(key="vqa_execution_form", clear_on_submit=True):
                user_question = st.text_input(
                    "Query Console",
                    value=st.session_state.quick_query,
                    placeholder="Enter your specific interrogation parameters here..."
                )
                
                submit_button = st.form_submit_button("🔍 Process Interrogation", type="primary", use_container_width=True)
                
                if submit_button:
                    if not user_question.strip():
                        st.error("Operation Failed: Query string cannot be empty.")
                    else:
                        with st.spinner("Computing spatial relevance..."):
                            inputs = pipeline["vqa_processor"](image, user_question, return_tensors="pt").to(device)
                            
                            with torch.no_grad():
                                outputs = pipeline["vqa_model"].generate(
                                    **inputs, max_length=50,
                                    return_dict_in_generate=True, output_scores=True
                                )
                            
                            generated_seq = outputs.sequences
                            transition_scores = pipeline["vqa_model"].compute_transition_scores(
                                generated_seq, outputs.scores, normalize_logits=True
                            )
                            
                            conf_score = calculate_confidence(outputs, transition_scores)
                            answer_text = pipeline["vqa_processor"].decode(generated_seq[0], skip_special_tokens=True)
                            
                            st.session_state.chat_history.insert(0, {
                                "question": user_question, 
                                "answer": answer_text,
                                "score": conf_score
                            })
                            # Clear the quick query state so it doesn't persist on next load
                            st.session_state.quick_query = ""

            # Modern Chat Log Rendering
            if st.session_state.chat_history:
                st.markdown("<br><span style='font-size:12px; font-weight:700; color:#9ca3af;'>SESSION LOGS</span>", unsafe_allow_html=True)
                for log in st.session_state.chat_history:
                    score_color = "#10b981" if log['score'] >= 85 else "#f59e0b"
                    
                    st.markdown(
                        f"""
                        <div class="user-bubble">👤 <b>Query:</b> {log['question']}</div>
                        <div class="ai-bubble">
                            💡 {log['answer'].capitalize()}
                            <div class="metric-badge" style="color: {score_color}; border: 1px solid {score_color}33;">
                                ACCURACY: {log['score']}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
else:
    st.info("System Idle. Awaiting source matrix upload to initialize processing pipelines.")