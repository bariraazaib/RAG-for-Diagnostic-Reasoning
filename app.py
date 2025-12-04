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
GEMINI_API_KEY = "AIzaSyCkwbqccRPTUd3zLqJ3A6WagcdDRsMJQCY"

class DataExtractor:
    def __init__(self):
        self.zip_path = "./data.zip"
        self.extracted_path = "./data_extracted"
        self.github_url = "https://github.com/bariraazaib/RAG-for-Diagnostic-Reasoning/blob/main/mimic-iv-ext-direct-1.0.zip"
        
    def download_from_github(self):
        """Download ZIP file from GitHub"""
        try:
            st.info("📥 Downloading data from GitHub...")
            
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
                                status_text.text(f"Downloaded {downloaded}/{total_size} bytes")
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Successfully downloaded data from GitHub")
                return True
            else:
                st.error(f"❌ Failed to download file. HTTP Status: {response.status_code}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error downloading from GitHub: {e}")
            return False
        
    def extract_data(self):
        """Extract data from ZIP file"""
        # First, download the file if it doesn't exist
        if not os.path.exists(self.zip_path):
            if not self.download_from_github():
                return False
            
        try:
            # Create extraction directory
            os.makedirs(self.extracted_path, exist_ok=True)
            
            # Extract ZIP file
            st.info("📦 Extracting ZIP file...")
            
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
                    status_text.text(f"Extracting files... {i+1}/{total_files}")
                
                progress_bar.empty()
                status_text.empty()
            
            st.success("✅ Successfully extracted data from ZIP file")
            return True
            
        except Exception as e:
            st.error(f"❌ Error extracting ZIP file: {e}")
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
            st.info(f"📁 Knowledge graph path: {self.kg_path}")
        if self.cases_path:
            st.info(f"📁 Cases path: {self.cases_path}")
    
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

        st.info(f"📊 Found {kg_count} knowledge files and {case_count} case files")
        return kg_count, case_count

    def extract_knowledge(self):
        """Extract knowledge from KG files"""
        chunks = []

        if not self.kg_path or not os.path.exists(self.kg_path):
            st.error(f"❌ Knowledge graph path not found")
            st.info(f"💡 Checked paths: {self.possible_kg_paths}")
            return chunks

        # Set up progress
        files = [f for f in os.listdir(self.kg_path) if f.endswith('.json')]
        total_files = len(files)
        
        if total_files == 0:
            st.warning("⚠️ No JSON files found in knowledge graph directory")
            return chunks
            
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
                status_text.text(f"Processing knowledge files... {i+1}/{total_files}")
                
            except Exception as e:
                st.warning(f"⚠️ Error processing {filename}: {e}")
                continue

        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Extracted {len(chunks)} knowledge chunks from {total_files} files")
        return chunks

    def extract_patient_cases(self):
        """Extract patient cases and reasoning"""
        chunks = []

        if not self.cases_path or not os.path.exists(self.cases_path):
            st.error(f"❌ Cases path not found")
            st.info(f"💡 Checked paths: {self.possible_case_paths}")
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
        progress_bar = st.progress(0)
        status_text = st.empty()

        processed_files = 0
        for file_path, condition_folder in file_paths:
            self._process_case_file(file_path, condition_folder, chunks)
            processed_files += 1
            
            # Update progress
            progress = int(100 * processed_files / total_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing case files... {processed_files}/{total_files}")

        progress_bar.empty()
        status_text.empty()

        narratives = len([c for c in chunks if c['metadata']['type'] == 'narrative'])
        reasoning = len([c for c in chunks if c['metadata']['type'] == 'reasoning'])
        st.success(f"✅ Extracted {narratives} narrative chunks and {reasoning} reasoning chunks from {total_files} case files")
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
        st.info("🚀 Starting data extraction...")

        # Check if data exists
        kg_exists, cases_exists = self.check_data_exists()
        if not kg_exists and not cases_exists:
            st.error("❌ No valid data found after extraction.")
            st.info("💡 Please check the ZIP file structure")
            return []

        # Count files
        kg_count, case_count = self.count_files()

        if kg_count == 0 and case_count == 0:
            st.error("❌ No JSON files found in data directories.")
            return []

        # Extract data
        knowledge_chunks = self.extract_knowledge()
        case_chunks = self.extract_patient_cases()

        all_chunks = knowledge_chunks + case_chunks

        if all_chunks:
            st.success(f"🎯 Extraction complete: {len(knowledge_chunks)} knowledge + {len(case_chunks)} cases = {len(all_chunks)} total chunks")
        else:
            st.error("❌ No data chunks were extracted")

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

            st.success("✅ Created ChromaDB collections")
        except Exception as e:
            st.error(f"Error creating collections: {e}")

    def index_data(self):
        """Index all chunks into ChromaDB"""
        knowledge_docs, knowledge_metas, knowledge_ids = [], [], []
        case_docs, case_metas, case_ids = [], [], []

        try:
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
                status_text.text(f"Indexing chunks... {i+1}/{total_chunks}")

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
    st.set_page_config(
        page_title="Medical AI Assistant",
        page_icon="🏥",
        layout="wide"
    )

    # Simple and clean CSS
    st.markdown("""
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        /* Main styling */
        .stApp {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Header */
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            color: white;
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Cards */
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        /* Buttons enhancement */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Question area */
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            font-size: 1rem;
        }
        
        .stTextArea textarea:focus {
            border-color: #667eea;
        }
        </style>
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

    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🏥 Medical AI Assistant</h1>
            <p>AI-Powered Medical Diagnosis & Knowledge System</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    st.sidebar.header("⚙️ System Configuration")
    
    # Show API key status
    st.sidebar.success("🔑 API key configured")
    
    # Data extraction section
    st.sidebar.subheader("📁 Data Setup")
    
    if not st.session_state.data_extracted:
        if st.sidebar.button("📥 Download & Extract Data", type="primary"):
            with st.spinner("Downloading and extracting data..."):
                extractor = DataExtractor()
                if extractor.extract_data():
                    st.session_state.data_extracted = True
                    st.session_state.extractor = extractor
                    st.rerun()

    # Initialize system
    if st.session_state.data_extracted and not st.session_state.initialized:
        if st.sidebar.button("🚀 Initialize System", type="primary"):
            try:
                with st.spinner("🚀 Processing medical data and setting up RAG system..."):
                    # Initialize processor and extract data
                    processor = SimpleDataProcessor(st.session_state.extractor.extracted_path)
                    chunks = processor.run()

                    if not chunks:
                        st.error("❌ No data was extracted. Please check your data file structure.")
                        return

                    # Initialize RAG system
                    rag_system = SimpleRAGSystem(chunks)
                    rag_system.create_collections()
                    rag_system.index_data()

                    # Initialize Medical AI
                    st.session_state.medical_ai = MedicalAI(rag_system, GEMINI_API_KEY)
                    st.session_state.rag_system = rag_system
                    st.session_state.initialized = True

                st.success("✅ System initialized successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")

    # Main interface
    if st.session_state.initialized and st.session_state.medical_ai:
        st.header("💬 Ask Your Medical Question")

        # Question input
        question = st.text_area(
            "Enter your medical question:",
            placeholder="e.g., What are the symptoms of migraine? How is chest pain evaluated?",
            height=120
        )

        col1, col2 = st.columns([3, 1])
        
        with col1:
            ask_button = st.button("🔍 Get Medical Answer", type="primary", use_container_width=True)
        
        with col2:
            show_context = st.checkbox("📚 Show context")

        if ask_button and question:
            with st.spinner("🔍 Analyzing medical context..."):
                try:
                    # Get answer
                    answer = st.session_state.medical_ai.ask(question)

                    # Display answer
                    st.markdown("---")
                    st.subheader("🤖 Medical Answer")
                    st.info(f"**Question:** {question}")
                    st.success(answer)

                    # Show context if requested
                    if show_context:
                        st.subheader("📚 Retrieved Context")
                        context_chunks = st.session_state.rag_system.query(question, top_k=5)
                        
                        for i, chunk in enumerate(context_chunks):
                            with st.expander(f"Context Source {i+1}"):
                                st.text(chunk)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # Example questions
        st.markdown("---")
        st.subheader("💡 Example Questions")
        
        examples = [
            "What are the diagnostic criteria for migraine?",
            "How is chest pain evaluated in emergency settings?",
            "What are common risk factors for gastrointestinal bleeding?",
            "Describe the symptoms and diagnosis process for pneumonia",
            "What are the treatment options for asthma?",
            "How to diagnose and manage diabetes?"
        ]

        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"ex_{i}", use_container_width=True):
                    st.session_state.example_question = example
                    st.rerun()

        # System info in sidebar
        if st.session_state.rag_system:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 System Stats")
            
            knowledge_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'knowledge'])
            narrative_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'narrative'])
            reasoning_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'reasoning'])
            
            st.sidebar.metric("Knowledge", knowledge_count)
            st.sidebar.metric("Narratives", narrative_count)
            st.sidebar.metric("Reasoning", reasoning_count)
            st.sidebar.metric("Total", len(st.session_state.rag_system.chunks))

    else:
        # Welcome screen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="info-card">
                    <h3 style="text-align: center;">🧠 AI-Powered</h3>
                    <p style="text-align: center;">Advanced RAG system with Gemini AI</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="info-card">
                    <h3 style="text-align: center;">📚 Rich Database</h3>
                    <p style="text-align: center;">Thousands of medical cases</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="info-card">
                    <h3 style="text-align: center;">⚡ Fast & Accurate</h3>
                    <p style="text-align: center;">Quick medical insights</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("""
        👋 **Welcome! Get Started:**
        
        1. 📥 Click **'Download & Extract Data'** in sidebar
        2. 🚀 Click **'Initialize System'** to build RAG
        3. 💬 Start asking medical questions!
        
        *Data source: MIMIC-IV Clinical Database*
        """)

if __name__ == "__main__":
    main()
