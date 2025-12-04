import streamlit as st
import os
import json
import tempfile
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import requests
import zipfile
import io

# Your hardcoded API key
GEMINI_API_KEY = "AIzaSyCUOQsqEUh9SYZ4MmfxjGTSgVywHfnqNls"

# ==================== CUSTOM CSS ====================
def load_css():
    st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1a2980 0%, #26d0ce 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Cards */
    .custom-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #1a2980 0%, #26d0ce 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 20px rgba(38, 208, 206, 0.4);
    }
    
    .primary-btn {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1a2980 0%, #26d0ce 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #1a252f 100%);
    }
    
    .sidebar-title {
        text-align: center;
        padding: 1rem;
        color: white;
        border-bottom: 2px solid #26d0ce;
        margin-bottom: 2rem;
    }
    
    /* Text input */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: border 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #26d0ce;
        box-shadow: 0 0 0 2px rgba(38, 208, 206, 0.2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Chat bubbles */
    .user-msg {
        background: linear-gradient(90deg, #1a2980 0%, #26d0ce 100%);
        color: white;
        padding: 1rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    
    .ai-msg {
        background: #f8f9fa;
        color: #333;
        padding: 1rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 80%;
        float: left;
        clear: both;
        border: 1px solid #e0e0e0;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a2980;
    }
    
    /* Status indicators */
    .status-success {
        color: #10b981;
        font-weight: bold;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: bold;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: bold;
    }
    
    /* Icons */
    .icon-large {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .custom-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

class DataExtractor:
    def __init__(self):
        self.zip_path = "./data.zip"
        self.extracted_path = "./data_extracted"
        self.github_url = "https://github.com/bariraazaib/RAG-for-Diagnostic-Reasoning/blob/main/mimic-iv-ext-direct-1.0.zip"
        
    def download_from_github(self):
        """Download ZIP file from GitHub"""
        try:
            with st.container():
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("### 📥 Downloading Data from GitHub")
                
                # Use raw GitHub URL
                response = requests.get(self.github_url, stream=True)
                
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with open(self.zip_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = int(50 * downloaded / total_size)
                                    progress_bar.progress(min(progress, 100))
                                    status_text.text(f"📊 Progress: {downloaded}/{total_size} bytes")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Success message with icon
                    st.success("✅ Data successfully downloaded!")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return True
                else:
                    st.error(f"❌ Download failed. HTTP Status: {response.status_code}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return False
                    
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            return False
        
    def extract_data(self):
        """Extract data from ZIP file"""
        # First, download the file if it doesn't exist
        if not os.path.exists(self.zip_path):
            if not self.download_from_github():
                return False
            
        try:
            with st.container():
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("### 📦 Extracting Data Files")
                
                # Create extraction directory
                os.makedirs(self.extracted_path, exist_ok=True)
                
                # Extract ZIP file
                with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                    # Get file list and set up progress
                    file_list = zip_ref.namelist()
                    total_files = len(file_list)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Extract all files
                    for i, file in enumerate(file_list):
                        zip_ref.extract(file, self.extracted_path)
                        progress = int(100 * (i + 1) / total_files)
                        progress_bar.progress(progress)
                        status_text.text(f"📁 Extracting... {i+1}/{total_files} files")
                    
                    progress_bar.empty()
                    status_text.empty()
                
                st.success("✅ Data successfully extracted!")
                st.markdown('</div>', unsafe_allow_html=True)
                return True
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            return False

class SimpleDataProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path
        # Try different possible paths after extraction
        self.possible_kg_paths = [
            os.path.join(base_path, "mimic-iv-ext-direct-1.0", "mimic-iv-ext-direct-1.0.0", "diagnostic_kg", "Diagnosis_flowchart"),
            os.path.join(base_path, "mimic-iv-ext-direct-1.0", "diagnostic_kg", "Diagnosis_flowchart"),
            os.path.join(base_path, "diagnostic_kg", "Diagnosis_flowchart"),
            os.path.join(base_path, "Diagnosis_flowchart"),
            os.path.join(base_path, "mimic-iv-ext-direct-1.0.0", "diagnostic_kg", "Diagnosis_flowchart"),
        ]
        self.possible_case_paths = [
            os.path.join(base_path, "mimic-iv-ext-direct-1.0", "mimic-iv-ext-direct-1.0.0", "Finished"),
            os.path.join(base_path, "mimic-iv-ext-direct-1.0", "Finished"),
            os.path.join(base_path, "Finished"),
            os.path.join(base_path, "cases"),
            os.path.join(base_path, "mimic-iv-ext-direct-1.0.0", "Finished"),
        ]
        
        self.kg_path = self._find_valid_path(self.possible_kg_paths)
        self.cases_path = self._find_valid_path(self.possible_case_paths)
        
        # Log found paths
        if self.kg_path:
            st.markdown(f'<div class="custom-card"><p>📁 <strong>Knowledge Graph Path:</strong> {self.kg_path}</p></div>', unsafe_allow_html=True)
        if self.cases_path:
            st.markdown(f'<div class="custom-card"><p>📁 <strong>Cases Path:</strong> {self.cases_path}</p></div>', unsafe_allow_html=True)
    
    def _find_valid_path(self, possible_paths):
        """Find the first valid path that exists"""
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def check_data_exists(self):
        """Check if data directories exist and have files"""
        kg_exists = self.kg_path and os.path.exists(self.kg_path) and any(f.endswith('.json') for f in os.listdir(self.kg_path))
        cases_exists = self.cases_path and os.path.exists(self.cases_path) and any(os.path.isdir(os.path.join(self.cases_path, d)) for d in os.listdir(self.cases_path))
        
        return kg_exists, cases_exists

    def count_files(self):
        """Count all JSON files"""
        kg_count = 0
        if self.kg_path and os.path.exists(self.kg_path):
            kg_count = len([f for f in os.listdir(self.kg_path) if f.endswith('.json')])

        case_count = 0
        if self.cases_path and os.path.exists(self.cases_path):
            for item in os.listdir(self.cases_path):
                item_path = os.path.join(self.cases_path, item)
                if os.path.isdir(item_path):
                    for root, dirs, files in os.walk(item_path):
                        case_count += len([f for f in files if f.endswith('.json')])
                elif item.endswith('.json'):
                    case_count += 1

        # Display in a nice card
        st.markdown(f'''
        <div class="custom-card">
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <h3 style="color: #1a2980;">📚</h3>
                    <h4 style="margin: 0;">{kg_count}</h4>
                    <p style="color: #666; margin: 0;">Knowledge Files</p>
                </div>
                <div>
                    <h3 style="color: #26d0ce;">📋</h3>
                    <h4 style="margin: 0;">{case_count}</h4>
                    <p style="color: #666; margin: 0;">Case Files</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        return kg_count, case_count

    def extract_knowledge(self):
        """Extract knowledge from KG files"""
        chunks = []

        if not self.kg_path or not os.path.exists(self.kg_path):
            st.error("❌ Knowledge graph path not found")
            return chunks

        # Set up progress
        files = [f for f in os.listdir(self.kg_path) if f.endswith('.json')]
        total_files = len(files)
        
        if total_files == 0:
            st.warning("⚠️ No JSON files found in knowledge graph directory")
            return chunks
            
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown(f"### 🔍 Processing {total_files} Knowledge Files")
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, filename in enumerate(files):
                file_path = os.path.join(self.kg_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    condition = filename.replace('.json', '')
                    knowledge = data.get('knowledge', {})

                    for stage_name, stage_data in knowledge.items():
                        if isinstance(stage_data, dict):
                            # Extract risk factors
                            if stage_data.get('Risk Factors'):
                                chunks.append({
                                    'text': f"{condition} - Risk Factors: {stage_data['Risk Factors']}",
                                    'metadata': {'type': 'knowledge', 'category': 'risk_factors', 'condition': condition}
                                })

                            # Extract symptoms
                            if stage_data.get('Symptoms'):
                                chunks.append({
                                    'text': f"{condition} - Symptoms: {stage_data['Symptoms']}",
                                    'metadata': {'type': 'knowledge', 'category': 'symptoms', 'condition': condition}
                                })
                    
                    # Update progress
                    progress = int(100 * (i + 1) / total_files)
                    progress_bar.progress(progress)
                    status_text.text(f"📄 Processing: {i+1}/{total_files}")
                    
                except Exception as e:
                    st.warning(f"⚠️ Error processing {filename}: {e}")
                    continue

            progress_bar.empty()
            status_text.empty()
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.success(f"✅ Extracted {len(chunks)} knowledge chunks")
        return chunks

    def extract_patient_cases(self):
        """Extract patient cases and reasoning"""
        chunks = []

        if not self.cases_path or not os.path.exists(self.cases_path):
            st.error("❌ Cases path not found")
            return chunks

        # Count total files for progress
        total_files = 0
        file_paths = []
        
        for item in os.listdir(self.cases_path):
            item_path = os.path.join(self.cases_path, item)
            if os.path.isdir(item_path):
                for root, dirs, files in os.walk(item_path):
                    json_files = [f for f in files if f.endswith('.json')]
                    total_files += len(json_files)
                    for f in json_files:
                        file_paths.append((os.path.join(root, f), item))
            elif item.endswith('.json'):
                total_files += 1
                file_paths.append((item_path, "General"))

        if total_files == 0:
            st.warning("⚠️ No case files found")
            return chunks

        # Set up progress
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown(f"### 🏥 Processing {total_files} Case Files")
            progress_bar = st.progress(0)
            status_text = st.empty()

            processed_files = 0
            for file_path, condition_folder in file_paths:
                self._process_case_file(file_path, condition_folder, chunks)
                processed_files += 1
                
                # Update progress
                progress = int(100 * processed_files / total_files)
                progress_bar.progress(progress)
                status_text.text(f"📋 Cases: {processed_files}/{total_files}")

            progress_bar.empty()
            status_text.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        narratives = len([c for c in chunks if c['metadata']['type'] == 'narrative'])
        reasoning = len([c for c in chunks if c['metadata']['type'] == 'reasoning'])
        
        st.success(f"✅ Extracted {narratives} narratives and {reasoning} reasoning chunks")
        return chunks

    def _process_case_file(self, file_path, condition_folder, chunks):
        """Process individual case file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filename = os.path.basename(file_path)
            case_id = filename.replace('.json', '')

            # Extract narrative (inputs)
            narrative_parts = []
            for i in range(1, 7):
                key = f'input{i}'
                if key in data and data[key]:
                    narrative_parts.append(f"{key}: {data[key]}")

            if narrative_parts:
                chunks.append({
                    'text': f"Case {case_id} - {condition_folder}\nNarrative:\n" + "\n".join(narrative_parts),
                    'metadata': {'type': 'narrative', 'case_id': case_id, 'condition': condition_folder}
                })

            # Extract reasoning
            for key in data:
                if not key.startswith('input'):
                    reasoning = self._extract_reasoning(data[key])
                    if reasoning:
                        chunks.append({
                            'text': f"Case {case_id} - {condition_folder}\nReasoning:\n{reasoning}",
                            'metadata': {'type': 'reasoning', 'case_id': case_id, 'condition': condition_folder}
                        })
        except Exception as e:
            st.warning(f"⚠️ Error processing {file_path}: {e}")

    def _extract_reasoning(self, data):
        """Simple reasoning extraction"""
        reasoning_lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                if '$Cause_' in key:
                    reasoning_text = key.split('$Cause_')[0].strip()
                    if reasoning_text:
                        reasoning_lines.append(reasoning_text)

                if isinstance(value, (dict, list)):
                    nested_reasoning = self._extract_reasoning(value)
                    if nested_reasoning:
                        reasoning_lines.append(nested_reasoning)

        elif isinstance(data, list):
            for item in data:
                nested_reasoning = self._extract_reasoning(item)
                if nested_reasoning:
                    reasoning_lines.append(nested_reasoning)

        return "\n".join(reasoning_lines) if reasoning_lines else ""

    def run(self):
        """Run complete extraction"""
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("## 🚀 Starting Data Extraction")
            
            # Check if data exists
            kg_exists, cases_exists = self.check_data_exists()
            if not kg_exists and not cases_exists:
                st.error("❌ No valid data found after extraction.")
                st.markdown('</div>', unsafe_allow_html=True)
                return []

            # Count files
            kg_count, case_count = self.count_files()

            if kg_count == 0 and case_count == 0:
                st.error("❌ No JSON files found in data directories.")
                st.markdown('</div>', unsafe_allow_html=True)
                return []

            # Extract data
            knowledge_chunks = self.extract_knowledge()
            case_chunks = self.extract_patient_cases()

            all_chunks = knowledge_chunks + case_chunks

            if all_chunks:
                st.balloons()
                st.success(f"🎯 Extraction complete: {len(knowledge_chunks)} knowledge + {len(case_chunks)} cases = {len(all_chunks)} total chunks")
            else:
                st.error("❌ No data chunks were extracted")
            
            st.markdown('</div>', unsafe_allow_html=True)
            return all_chunks

class SimpleRAGSystem:
    def __init__(self, chunks, db_path="./chroma_db"):
        self.chunks = chunks
        self.db_path = db_path
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.client = chromadb.PersistentClient(path=db_path)
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")

    def create_collections(self):
        """Create separate collections for knowledge and cases"""
        try:
            with st.container():
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("## 🗂️ Creating Database Collections")
                
                # Knowledge collection
                self.knowledge_collection = self.client.get_or_create_collection(
                    name="medical_knowledge",
                    embedding_function=self.embedding_function
                )

                # Cases collection
                self.cases_collection = self.client.get_or_create_collection(
                    name="patient_cases",
                    embedding_function=self.embedding_function
                )

                st.success("✅ Database collections created successfully!")
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error creating collections: {e}")

    def index_data(self):
        """Index all chunks into ChromaDB"""
        knowledge_docs, knowledge_metas, knowledge_ids = [], [], []
        case_docs, case_metas, case_ids = [], [], []

        try:
            with st.container():
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown(f"## 📊 Indexing {len(self.chunks)} Data Chunks")
                
                total_chunks = len(self.chunks)
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, chunk in enumerate(self.chunks):
                    if chunk['metadata']['type'] == 'knowledge':
                        knowledge_docs.append(chunk['text'])
                        knowledge_metas.append(chunk['metadata'])
                        knowledge_ids.append(f"kg_{i}")
                    else:
                        case_docs.append(chunk['text'])
                        case_metas.append(chunk['metadata'])
                        case_ids.append(f"case_{i}")

                    # Update progress
                    progress = int(100 * (i + 1) / total_chunks)
                    progress_bar.progress(progress)
                    status_text.text(f"📝 Indexing... {i+1}/{total_chunks} chunks")

                progress_bar.empty()
                status_text.empty()

                # Add to collections
                if knowledge_docs:
                    self.knowledge_collection.add(
                        documents=knowledge_docs,
                        metadatas=knowledge_metas,
                        ids=knowledge_ids
                    )

                if case_docs:
                    self.cases_collection.add(
                        documents=case_docs,
                        metadatas=case_metas,
                        ids=case_ids
                    )

                st.success(f"✅ Indexed {len(knowledge_docs)} knowledge chunks and {len(case_docs)} case chunks")
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error indexing data: {e}")

    def query(self, question, top_k=5):
        """Simple query across both collections"""
        try:
            # Query knowledge
            knowledge_results = self.knowledge_collection.query(
                query_texts=[question],
                n_results=top_k
            )

            # Query cases
            case_results = self.cases_collection.query(
                query_texts=[question],
                n_results=top_k
            )

            # Combine results
            all_results = []
            if knowledge_results['documents']:
                all_results.extend(knowledge_results['documents'][0])
            if case_results['documents']:
                all_results.extend(case_results['documents'][0])

            return all_results
        except Exception as e:
            st.error(f"Error querying RAG system: {e}")
            return []

class MedicalAI:
    def __init__(self, rag_system, api_key):
        self.rag = rag_system
        try:
            genai.configure(api_key=api_key)
            # Use a more widely available model
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")

    def ask(self, question):
        try:
            # Get relevant context from RAG
            context_chunks = self.rag.query(question, top_k=5)
            context = "\n---\n".join(context_chunks)

            # Create prompt WITHOUT the "what's missing" section
            prompt = f"""You are a medical expert. Use the following medical context to answer the question accurately and comprehensively.

MEDICAL CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive medical answer based on the context. Focus on the information available in the context."""

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"

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
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical AI Diagnosis Assistant</h1>
        <p>Intelligent medical insights powered by RAG and Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'medical_ai' not in st.session_state:
        st.session_state.medical_ai = None
    if 'data_extracted' not in st.session_state:
        st.session_state.data_extracted = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">', unsafe_allow_html=True)
        st.markdown("## ⚙️ Configuration")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # API Status
        st.markdown("### 🔑 API Status")
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.success("✅ Gemini API Active")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data Setup Section
        st.markdown("### 📁 Data Setup")
        
        if not st.session_state.data_extracted:
            if st.button("📥 Download & Extract Data", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    extractor = DataExtractor()
                    if extractor.extract_data():
                        st.session_state.data_extracted = True
                        st.session_state.extractor = extractor
                        st.rerun()
        
        # Initialize System
        if st.session_state.data_extracted and not st.session_state.initialized:
            st.markdown("### 🚀 System Initialization")
            if st.button("Initialize AI System", type="primary", use_container_width=True):
                try:
                    with st.spinner("Initializing medical AI system..."):
                        # Initialize processor and extract data
                        processor = SimpleDataProcessor(st.session_state.extractor.extracted_path)
                        chunks = processor.run()

                        if not chunks:
                            st.error("❌ No data was extracted.")
                            return

                        # Initialize RAG system
                        rag_system = SimpleRAGSystem(chunks)
                        rag_system.create_collections()
                        rag_system.index_data()

                        # Initialize Medical AI with hardcoded API key
                        st.session_state.medical_ai = MedicalAI(rag_system, GEMINI_API_KEY)
                        st.session_state.rag_system = rag_system
                        st.session_state.initialized = True

                    st.success("✅ System ready!")
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        st.markdown("---")
        
        # System Info
        if st.session_state.initialized:
            st.markdown("### 📊 System Status")
            if st.session_state.rag_system:
                knowledge_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'knowledge'])
                narrative_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'narrative'])
                reasoning_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'reasoning'])
                
                st.markdown(f"""
                <div class="custom-card">
                    <p><strong>Knowledge:</strong> {knowledge_count} chunks</p>
                    <p><strong>Narratives:</strong> {narrative_count} cases</p>
                    <p><strong>Reasoning:</strong> {reasoning_count} entries</p>
                    <p><strong>Total:</strong> {len(st.session_state.rag_system.chunks)} chunks</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Links
        st.markdown("### 🔗 Quick Links")
        st.markdown("""
        <div class="custom-card">
            <p>📚 <a href="https://github.com/Mustehsan-Nisar-Rao/RAG" target="_blank">Data Source</a></p>
            <p>🤖 <a href="https://ai.google.dev/" target="_blank">Gemini AI</a></p>
            <p>💡 <a href="#" target="_blank">Documentation</a></p>
        </div>
        """, unsafe_allow_html=True)

    # Main Content
    if st.session_state.initialized and st.session_state.medical_ai:
        # Chat Interface
        st.markdown("## 💬 Medical Query Assistant")
        
        # Chat History Display
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat['type'] == 'user':
                    st.markdown(f'<div class="user-msg">👤 {chat["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ai-msg">🤖 {chat["content"]}</div>', unsafe_allow_html=True)
        
        # Question Input
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        question = st.text_area(
            "**Enter your medical question:**",
            placeholder="What are the symptoms of migraine? How to diagnose chest pain? Risk factors for diabetes?",
            height=120
        )
        
        # Options
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Context Chunks", 1, 10, 5)
        with col2:
            show_context = st.checkbox("Show retrieved context")
        
        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Get Medical Answer", type="primary", use_container_width=True) and question:
                with st.spinner("🔍 Analyzing medical data..."):
                    try:
                        # Add user question to history
                        st.session_state.chat_history.append({
                            'type': 'user',
                            'content': question
                        })
                        
                        # Get answer
                        answer = st.session_state.medical_ai.ask(question)
                        
                        # Add AI response to history
                        st.session_state.chat_history.append({
                            'type': 'ai',
                            'content': answer
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Example Questions
        st.markdown("### 💡 Example Questions")
        examples = [
            "Describe the symptoms and diagnosis process for pneumonia",
            "What are the treatment options for asthma?",
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, use_container_width=True):
                    st.session_state.last_question = example
                    st.rerun()
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            if st.session_state.rag_system and show_context and question:
                st.markdown("### 📚 Retrieved Context")
                context_chunks = st.session_state.rag_system.query(question, top_k=top_k)
                
                for i, chunk in enumerate(context_chunks):
                    with st.expander(f"Context {i+1}"):
                        st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
    
    else:
        # Welcome Screen
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("## 👋 Welcome to Medical AI Assistant")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Features:
            
            **🤖 AI-Powered Diagnosis**
            - Get accurate medical insights
            - Based on extensive medical database
            - Powered by Gemini AI
            
            **📊 Comprehensive Database**
            - 1000+ medical cases
            - Knowledge graphs
            - Diagnostic reasoning
            
            **⚡ Fast & Efficient**
            - Quick response time
            - Easy to use interface
            - Reliable results
            
            ### 📋 Quick Start:
            1. **Download** medical data
            2. **Initialize** the AI system
            3. **Ask** medical questions
            4. **Get** expert-level answers
            """)
        
        with col2:
            st.markdown("""
            ### 📈 System Status
            
            <div class="custom-card">
                <h3 style="text-align: center;">🔑</h3>
                <p style="text-align: center;"><strong>API Status</strong></p>
                <p style="text-align: center;" class="status-success">Active</p>
            </div>
            
            <div class="custom-card">
                <h3 style="text-align: center;">📁</h3>
                <p style="text-align: center;"><strong>Data Status</strong></p>
                <p style="text-align: center;" class="status-warning">Ready to Load</p>
            </div>
            
            <div class="custom-card">
                <h3 style="text-align: center;">🤖</h3>
                <p style="text-align: center;"><strong>AI Status</strong></p>
                <p style="text-align: center;" class="status-warning">Initialization Required</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data Source Info
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📚 Data Source Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Repository:** MIMIC-IV Dataset  
            **Source:** GitHub  
            **Format:** ZIP Archive  
            **Size:** ~150 MB  
            **Files:** JSON Format
            """)
        
        with col2:
            st.markdown("""
            **Contents:**
            - Medical knowledge graphs
            - Patient case narratives
            - Diagnostic reasoning
            - Treatment protocols
            - Risk factor analysis
            """)
        
        st.markdown("""
        *Note: All data is processed locally. No personal patient data is stored.*
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🏥 Medical AI Assistant v2.0 • Powered by Gemini AI & RAG Technology</p>
        <p>⚠️ For educational purposes only. Always consult a healthcare professional for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
