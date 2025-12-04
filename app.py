# frontend.py - Modern Streamlit UI
import streamlit as st
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import time

# ==================== CUSTOM CSS ====================
def load_css():
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Cards */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50, #1a252f);
        color: white;
    }
    
    /* Text input */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 15px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        background: #f0f2f6;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px;
        max-width: 80%;
        float: right;
    }
    
    .ai-bubble {
        background: #f1f3f9;
        color: #333;
        padding: 15px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px;
        max-width: 80%;
        float: left;
        border: 1px solid #e0e0e0;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOTTIE ANIMATIONS ====================
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ==================== DASHBOARD COMPONENTS ====================
def create_metric_card(title, value, delta=None, icon="📊"):
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"<h1 style='font-size: 3rem;'>{icon}</h1>", unsafe_allow_html=True)
    with col2:
        st.metric(label=title, value=value, delta=delta)

def create_progress_card(title, value, max_value=100, color="#667eea"):
    progress = (value / max_value) * 100
    st.markdown(f"""
    <div class="card">
        <h4>{title}</h4>
        <div style="background: #e0e0e0; border-radius: 10px; height: 20px; margin: 10px 0;">
            <div style="background: {color}; width: {progress}%; height: 100%; border-radius: 10px;"></div>
        </div>
        <p style="text-align: right; margin: 0;">{value}/{max_value}</p>
    </div>
    """, unsafe_allow_html=True)

def create_chat_bubble(message, is_user=True):
    if is_user:
        st.markdown(f"""
        <div class="user-bubble">
            👤 {message}
        </div>
        <div style="clear: both;"></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ai-bubble">
            🤖 {message}
        </div>
        <div style="clear: both;"></div>
        """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    # Page config
    st.set_page_config(
        page_title="Medical AI Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: white;">🏥 Medical AI</h1>
            <p style="color: #bbb;">Intelligent Diagnosis Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Chat Assistant", "Data Analysis", "Settings"],
            icons=["speedometer2", "robot", "bar-chart", "gear"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "white", "font-size": "20px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "10px 0",
                    "border-radius": "10px",
                    "color": "white",
                    "--hover-color": "#667eea",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                },
            }
        )
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 📊 System Status")
        
        status_cols = st.columns(2)
        with status_cols[0]:
            st.success("✅ Online")
        with status_cols[1]:
            st.info("🔑 API Active")
        
        # Progress indicators
        st.markdown("### 📈 Progress")
        create_progress_card("Data Extraction", 85, 100, "#4CAF50")
        create_progress_card("Indexing", 92, 100, "#2196F3")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.toast("Refreshing data...", icon="🔄")
            time.sleep(1)
        
        if st.button("📊 View Reports", use_container_width=True):
            st.toast("Generating reports...", icon="📊")
            time.sleep(1)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 12px;">
            <p>v2.1.0 • Medical AI Assistant</p>
            <p>© 2024 All rights reserved</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== MAIN CONTENT ====================
    if selected == "Dashboard":
        st.markdown("# 📊 Medical AI Dashboard")
        st.markdown("Welcome to your intelligent medical assistant")
        
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Total Cases", "1,247", "+12%", "📋")
        
        with col2:
            create_metric_card("Accuracy", "94.2%", "+2.1%", "🎯")
        
        with col3:
            create_metric_card("Response Time", "1.2s", "-0.3s", "⚡")
        
        with col4:
            create_metric_card("Active Users", "48", "+5", "👥")
        
        st.markdown("---")
        
        # Main content columns
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            # Data Processing Section
            st.markdown("### 📁 Data Processing")
            
            with st.expander("📥 Download Data", expanded=True):
                st.info("Click below to download medical data from GitHub repository")
                download_cols = st.columns([3, 1])
                with download_cols[0]:
                    st.markdown("**Source:** GitHub Repository")
                    st.markdown("**Size:** ~150 MB")
                with download_cols[1]:
                    if st.button("Download", key="download_btn"):
                        # Your existing download code here
                        with st.spinner("Downloading data..."):
                            time.sleep(2)
                            st.success("Data downloaded successfully!")
            
            with st.expander("🔧 Initialize System", expanded=True):
                st.info("Initialize the RAG system with downloaded data")
                init_cols = st.columns([3, 1])
                with init_cols[0]:
                    st.markdown("**Status:** Ready to initialize")
                    st.markdown("**Estimated time:** 2-3 minutes")
                with init_cols[1]:
                    if st.button("Initialize", key="init_btn"):
                        # Your existing initialization code here
                        progress_bar = st.progress(0)
                        for percent_complete in range(100):
                            time.sleep(0.03)
                            progress_bar.progress(percent_complete + 1)
                        st.success("System initialized successfully!")
            
            # Recent Activity
            st.markdown("### 📈 Recent Activity")
            
            activities = [
                {"time": "10:30 AM", "action": "System initialized", "user": "Admin", "status": "success"},
                {"time": "09:45 AM", "action": "Data indexed", "user": "System", "status": "success"},
                {"time": "09:15 AM", "action": "New query processed", "user": "Dr. Smith", "status": "info"},
                {"time": "08:30 AM", "action": "Database updated", "user": "System", "status": "warning"},
            ]
            
            for activity in activities:
                with st.container():
                    cols = st.columns([1, 3, 2, 1])
                    cols[0].markdown(f"**{activity['time']}**")
                    cols[1].markdown(activity['action'])
                    cols[2].markdown(f"👤 {activity['user']}")
                    if activity['status'] == 'success':
                        cols[3].success("✓")
                    elif activity['status'] == 'info':
                        cols[3].info("i")
                    else:
                        cols[3].warning("!")
                    st.markdown("---")
        
        with right_col:
            # System Health
            st.markdown("### 🏥 System Health")
            
            # Create health indicators
            health_data = {
                "Database": 95,
                "API Service": 98,
                "Memory Usage": 67,
                "Response Time": 92
            }
            
            for component, value in health_data.items():
                create_progress_card(component, value, 100)
            
            # Quick Stats
            st.markdown("### 📊 Quick Stats")
            
            stats_data = {
                "Daily Queries": 124,
                "Accuracy Rate": "94.2%",
                "Avg Response": "1.2s",
                "Active Sessions": 8
            }
            
            for key, val in stats_data.items():
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <p style="margin: 0; color: #666; font-size: 14px;">{key}</p>
                    <h3 style="margin: 5px 0; color: #667eea;">{val}</h3>
                </div>
                """, unsafe_allow_html=True)
    
    elif selected == "Chat Assistant":
        st.markdown("# 🤖 Medical Chat Assistant")
        st.markdown("Ask medical questions and get intelligent responses")
        
        # Chat container
        chat_container = st.container(height=500)
        
        with chat_container:
            # Sample chat history
            create_chat_bubble("What are the symptoms of migraine?", is_user=True)
            create_chat_bubble("Migraine symptoms typically include moderate to severe headache, nausea, vomiting, sensitivity to light and sound. Based on medical literature, common risk factors include...", is_user=False)
            
            create_chat_bubble("How to diagnose chest pain?", is_user=True)
            create_chat_bubble("Chest pain evaluation involves ECG, troponin levels, and clinical assessment. Our database shows 85% accuracy in diagnosing cardiac vs non-cardiac chest pain...", is_user=False)
        
        # Input area
        st.markdown("### 💬 Ask a Question")
        
        question = st.text_area(
            "Enter your medical question:",
            placeholder="Type your question here...",
            height=100,
            key="chat_input"
        )
        
        # Buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Get Answer", use_container_width=True):
                if question:
                    # Your existing RAG query code here
                    with st.spinner("Analyzing..."):
                        time.sleep(1)
                        create_chat_bubble(question, is_user=True)
                        create_chat_bubble("Based on medical literature, the symptoms include... [AI Response]", is_user=False)
                        st.rerun()
                else:
                    st.warning("Please enter a question")
        
        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("💾 Save Session", use_container_width=True):
                st.toast("Session saved!", icon="💾")
        
        with col4:
            if st.button("📋 Copy Answer", use_container_width=True):
                st.toast("Copied to clipboard!", icon="📋")
        
        # Example questions
        st.markdown("### 💡 Example Questions")
        
        examples = st.columns(3)
        example_questions = [
            "What are risk factors for diabetes?",
            "How to manage asthma?",
            "COVID-19 symptoms?",
            "Heart attack diagnosis",
            "Hypertension treatment",
            "Pediatric fever guidelines"
        ]
        
        for i, col in enumerate(examples):
            if i < len(example_questions):
                with col:
                    if st.button(example_questions[i], use_container_width=True):
                        st.session_state.last_question = example_questions[i]
                        st.rerun()
    
    elif selected == "Data Analysis":
        st.markdown("# 📈 Data Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📁 Data Structure", "🔍 Query Analysis"])
        
        with tab1:
            st.markdown("### Database Statistics")
            
            # Create sample charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                labels = ['Knowledge', 'Cases', 'Reasoning']
                values = [45, 35, 20]
                
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
                fig.update_layout(
                    title="Data Distribution",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart
                conditions = ['Cardiac', 'Neuro', 'GI', 'Respiratory', 'Other']
                counts = [120, 85, 65, 45, 30]
                
                fig = go.Figure(data=[go.Bar(x=conditions, y=counts, marker_color='#667eea')])
                fig.update_layout(
                    title="Cases by Condition",
                    height=300,
                    xaxis_title="Condition",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.markdown("### 📋 Sample Data")
            
            sample_data = {
                "Type": ["Knowledge", "Case", "Reasoning", "Knowledge", "Case"],
                "Content": ["Migraine symptoms...", "Patient case 123...", "Diagnostic reasoning...", "Diabetes risk...", "ECG analysis..."],
                "Source": ["Medical KG", "Case File", "Case File", "Medical KG", "Case File"],
                "Words": [150, 230, 180, 120, 195]
            }
            
            st.dataframe(sample_data, use_container_width=True)
        
        with tab2:
            st.markdown("### 📂 File Structure")
            
            # Tree view of data structure
            st.markdown("""
            ```
            data_extracted/
            ├── mimic-iv-ext-direct-1.0/
            │   ├── diagnostic_kg/
            │   │   └── Diagnosis_flowchart/
            │   │       ├── condition1.json
            │   │       ├── condition2.json
            │   │       └── ...
            │   └── Finished/
            │       ├── case1/
            │       │   ├── data.json
            │       │   └── notes.json
            │       └── case2/
            │           └── data.json
            └── metadata.json
            ```
            """)
            
            # File stats
            col1, col2, col3 = st.columns(3)
            col1.metric("JSON Files", "1,247")
            col2.metric("Total Size", "245 MB")
            col3.metric("Conditions", "48")
        
        with tab3:
            st.markdown("### 🔍 Query Performance")
            
            # Query metrics
            metrics = {
                "Avg Response Time": "1.2s",
                "Success Rate": "98.5%",
                "Cache Hit Rate": "67%",
                "API Calls": "1,248"
            }
            
            cols = st.columns(4)
            for i, (key, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(key, value)
            
            # Recent queries table
            st.markdown("### Recent Queries")
            
            queries = [
                {"Query": "Heart attack symptoms", "Response Time": "1.1s", "Accuracy": "95%"},
                {"Query": "Diabetes medication", "Response Time": "1.3s", "Accuracy": "92%"},
                {"Query": "Pediatric fever", "Response Time": "0.9s", "Accuracy": "96%"},
                {"Query": "COVID diagnosis", "Response Time": "1.5s", "Accuracy": "94%"},
            ]
            
            st.dataframe(queries, use_container_width=True)
    
    elif selected == "Settings":
        st.markdown("# ⚙️ Settings")
        
        with st.form("settings_form"):
            st.markdown("### API Configuration")
            
            # Read-only API key display
            api_key = st.text_input(
                "Gemini API Key",
                value="AIzaSyCkwbqccRPTUd3zLqJ3A6WagcdDRsMJQCY",
                type="password",
                disabled=True,
                help="API key is pre-configured for security"
            )
            
            st.markdown("### Model Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model = st.selectbox(
                    "Model",
                    ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                    index=0
                )
                
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1
                )
            
            with col2:
                max_tokens = st.slider(
                    "Max Tokens",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    step=100
                )
                
                top_k = st.slider(
                    "Top K (RAG)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1
                )
            
            st.markdown("### Data Settings")
            
            data_path = st.text_input(
                "Data Directory",
                value="./data_extracted",
                help="Path to extracted medical data"
            )
            
            auto_refresh = st.checkbox(
                "Auto-refresh data",
                value=True,
                help="Automatically check for data updates"
            )
            
            # Submit button
            if st.form_submit_button("💾 Save Settings", use_container_width=True):
                st.toast("Settings saved successfully!", icon="✅")
                time.sleep(1)

# Run the app
if __name__ == "__main__":
    main()
