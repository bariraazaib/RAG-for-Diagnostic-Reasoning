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

# Hardcoded API Key
GEMINI_API_KEY = "AIzaSyAHVnZccxmLFRDHGaOkZLBhRDw7kbcjFsM"

class DataExtractor:
    def __init__(self):
        self.zip_path = "./data.zip"
        self.extracted_path = "./data_extracted"
        self.github_url = "https://github.com/barirazaib/RAG/raw/main/mimic-iv-ext-direct-1.0.zip"
        
    def download_from_github(self):
        """Download ZIP file from GitHub"""
        try:
            st.info("📥 Downloading data from GitHub...")
            
            response = requests.get(self.github_url, stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                
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
        if not os.path.exists(self.zip_path):
            if not self.download_from_github():
                return False
            
        try:
            os.makedirs(self.extracted_path, exist_ok=True)
            
            st.info("📦 Extracting ZIP file...")
            
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
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
                        if stage_data.get('Risk Factors'):
                            chunks.append({
                                'text': f"{condition} - Risk Factors: {stage_data['Risk Factors']}",
                                'metadata': {'type': 'knowledge', 'category': 'risk_factors', 'condition': condition}
                            })

                        if stage_data.get('Symptoms'):
                            chunks.append({
                                'text': f"{condition} - Symptoms: {stage_data['Symptoms']}",
                                'metadata': {'type': 'knowledge', 'category': 'symptoms', 'condition': condition}
                            })
                
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

        progress_bar = st.progress(0)
        status_text = st.empty()

        processed_files = 0
        for file_path, condition_folder in file_paths:
            self._process_case_file(file_path, condition_folder, chunks)
            processed_files += 1
            
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

        kg_exists, cases_exists = self.check_data_exists()
        if not kg_exists and not cases_exists:
            st.error("❌ No valid data found after extraction.")
            st.info("💡 Please check the ZIP file structure")
            return []

        kg_count, case_count = self.count_files()

        if kg_count == 0 and case_count == 0:
            st.error("❌ No JSON files found in data directories.")
            return []

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
            self.knowledge_collection = self.client.get_or_create_collection(
                name="medical_knowledge",
                embedding_function=self.embedding_function
            )

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

                progress = int(100 * (i + 1) / total_chunks)
                progress_bar.progress(progress)
                status_text.text(f"Indexing chunks... {i+1}/{total_chunks}")

            progress_bar.empty()
            status_text.empty()

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
            knowledge_results = self.knowledge_collection.query(
                query_texts=[question],
                n_results=top_k
            )

            case_results = self.cases_collection.query(
                query_texts=[question],
                n_results=top_k
            )

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
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")

    def ask(self, question):
        try:
            context_chunks = self.rag.query(question, top_k=5)
            context = "\n---\n".join(context_chunks)

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
        page_title="MediAssist AI",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="header-container">
            <h1 style="margin:0; font-size: 3rem;">🩺 MediAssist AI</h1>
            <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">Advanced Medical Diagnosis & Knowledge Assistant</p>
        </div>
    """, unsafe_allow_html=True)

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

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/stethoscope.png", width=80)
        st.title("⚙️ Control Panel")
        
        st.success("✅ API Key Configured")
        
        st.divider()
        
        st.subheader("📦 Data Management")
        
        if not st.session_state.data_extracted:
            if st.button("📥 Download Medical Data", type="primary", use_container_width=True):
                with st.spinner("Downloading and extracting medical data..."):
                    extractor = DataExtractor()
                    if extractor.extract_data():
                        st.session_state.data_extracted = True
                        st.session_state.extractor = extractor
                        st.rerun()
        else:
            st.success("✅ Data Ready")

        if st.session_state.data_extracted and not st.session_state.initialized:
            st.divider()
            if st.button("🚀 Initialize AI System", type="primary", use_container_width=True):
                try:
                    with st.spinner("🔧 Building intelligent medical assistant..."):
                        processor = SimpleDataProcessor(st.session_state.extractor.extracted_path)
                        chunks = processor.run()

                        if not chunks:
                            st.error("❌ No data extracted. Check file structure.")
                            return

                        rag_system = SimpleRAGSystem(chunks)
                        rag_system.create_collections()
                        rag_system.index_data()

                        st.session_state.medical_ai = MedicalAI(rag_system, GEMINI_API_KEY)
                        st.session_state.rag_system = rag_system
                        st.session_state.initialized = True

                    st.success("✅ System Ready!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Initialization error: {str(e)}")

        if st.session_state.initialized and st.session_state.rag_system:
            st.divider()
            st.subheader("📊 System Statistics")
            
            knowledge_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'knowledge'])
            narrative_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'narrative'])
            reasoning_count = len([c for c in st.session_state.rag_system.chunks if c['metadata']['type'] == 'reasoning'])
            
            st.metric("Knowledge Base", f"{knowledge_count:,}")
            st.metric("Case Studies", f"{narrative_count:,}")
            st.metric("Reasoning Paths", f"{reasoning_count:,}")
            st.metric("Total Documents", f"{len(st.session_state.rag_system.chunks):,}")

        st.divider()
        st.caption("🔬 Powered by Gemini AI")
        st.caption("📚 Data: github.com/barirazaib/RAG")

    if st.session_state.initialized and st.session_state.medical_ai:
        
        tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📚 Example Queries", "ℹ️ About"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🔍 Medical Query Interface")
                
                question = st.text_area(
                    "Enter your medical question:",
                    placeholder="Describe symptoms, ask about conditions, or request diagnostic information...",
                    height=120,
                    key="question_input"
                )
                
                col_btn1, col_btn2 = st.columns([1, 3])
                with col_btn1:
                    submit = st.button("🔍 Analyze", type="primary", use_container_width=True)
                with col_btn2:
                    clear = st.button("🗑️ Clear History", use_container_width=True)
                
                if clear:
                    st.session_state.chat_history = []
                    st.rerun()

            with col2:
                with st.expander("⚙️ Advanced Settings", expanded=False):
                    top_k = st.slider("Context Depth", 1, 10, 5)
                    show_context = st.checkbox("Show Sources", value=False)

            if submit and question:
                with st.spinner("🧠 Analyzing medical data..."):
                    try:
                        answer = st.session_state.medical_ai.ask(question)
                        
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer
                        })
                        
                        st.markdown("---")
                        st.markdown("### 🤖 Medical Analysis")
                        st.info(f"**Question:** {question}")
                        st.success(f"**Answer:**\n\n{answer}")

                        if show_context:
                            with st.expander("📚 Retrieved Medical Context"):
                                context_chunks = st.session_state.rag_system.query(question, top_k=top_k)
                                
                                for i, chunk in enumerate(context_chunks):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(chunk[:400] + "..." if len(chunk) > 400 else chunk)
                                    st.divider()

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("📜 Consultation History")
                for i, item in enumerate(reversed(st.session_state.chat_history[-5:])):
                    with st.expander(f"Query {len(st.session_state.chat_history)-i}: {item['question'][:60]}..."):
                        st.markdown(f"**Q:** {item['question']}")
                        st.markdown(f"**A:** {item['answer']}")

        with tab2:
            st.subheader("💡 Example Medical Queries")
            st.markdown("Click any question below to quickly explore the system's capabilities:")
            
            examples = {
                "🫀 Cardiovascular": [
                    "What are the warning signs and risk factors for acute myocardial infarction?",
                    "Explain the differential diagnosis approach for chest pain in emergency settings"
                ],
                "🧠 Neurological": [
                    "Describe the clinical presentation and diagnosis of ischemic stroke",
                    "What are the key differences between migraine types and their treatment protocols?"
                ],
                "🫁 Respiratory": [
                    "What are the diagnostic criteria and management steps for acute respiratory distress?",
                    "How do you differentiate between bacterial and viral pneumonia clinically?"
                ]
            }
            
            for category, questions in examples.items():
                st.markdown(f"### {category}")
                for question in questions:
                    if st.button(question, key=question, use_container_width=True):
                        st.session_state.selected_question = question
                        st.rerun()
                st.markdown("")

        with tab3:
            st.subheader("ℹ️ About MediAssist AI")
            
            st.markdown("""
            ### 🎯 What is MediAssist AI?
            
            MediAssist AI is an advanced medical knowledge assistant powered by RAG technology and Gemini AI.
            
            ### ⚠️ Important Disclaimer
            
            **This system is for educational and informational purposes only.**
            
            - Not a substitute for professional medical advice
            - Always consult qualified healthcare providers
            - For emergencies, contact emergency services immediately
            """)

    else:
        st.markdown("""
            <div style='text-align: center; padding: 3rem;'>
                <img src='https://img.icons8.com/fluency/200/000000/medical-doctor.png' width='150'>
                <h2 style='color: #667eea; margin-top: 1rem;'>Welcome to MediAssist AI</h2>
                <p style='font-size: 1.2rem; color: #666; max-width: 600px; margin: 1rem auto;'>
                    Your intelligent companion for medical knowledge exploration
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
                <h2 style='color: #667eea; text-align: center;'>🚀 Get Started in 2 Simple Steps</h2>
                <br>
                <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
                    <div style='flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #667eea;'>
                        <h3>Step 1: Download Data 📥</h3>
                        <p>Click <b>"Download Medical Data"</b> in the sidebar to fetch the medical knowledge base from GitHub</p>
                    </div>
                    <div style='flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #764ba2;'>
                        <h3>Step 2: Initialize System 🚀</h3>
                        <p>Click <b>"Initialize AI System"</b> to build your intelligent medical assistant</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Features grid
        st.markdown("### ✨ What Makes MediAssist AI Special?")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            - 🎯 **Context-Aware Responses**: Understands complex medical terminology
            - 📊 **Evidence-Based**: Answers backed by medical literature
            - 🔄 **Multi-Specialty**: Covers all major medical specialties
            - ⚡ **Fast Retrieval**: Instant access to relevant information
            """)
        
        with feature_col2:
            st.markdown("""
            - 🧠 **Intelligent Analysis**: Powered by Google's Gemini AI
            - 📚 **Comprehensive Database**: Thousands of medical documents
            - 🔍 **Semantic Search**: Finds contextually relevant information
            - 💡 **Educational Focus**: Learn diagnostic reasoning patterns
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Disclaimer box
        st.warning("""
            ⚠️ **Medical Disclaimer**: This system is designed for educational and informational purposes only. 
            It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
            Always seek the advice of qualified healthcare providers with questions regarding medical conditions.
        """)
        
        st.info("""
            💡 **Tip**: Once initialized, you can ask questions about symptoms, diagnoses, treatment protocols, 
            and explore real patient case studies to enhance your medical knowledge.
        """)

    # Handle selected question from examples
    if hasattr(st.session_state, 'selected_question') and st.session_state.selected_question:
        if st.session_state.initialized and st.session_state.medical_ai:
            with st.spinner("🧠 Analyzing medical data..."):
                try:
                    answer = st.session_state.medical_ai.ask(st.session_state.selected_question)
                    
                    st.session_state.chat_history.append({
                        'question': st.session_state.selected_question,
                        'answer': answer
                    })
                    
                    st.markdown("---")
                    st.markdown("### 🤖 Medical Analysis")
                    st.info(f"**Question:** {st.session_state.selected_question}")
                    st.success(f"**Answer:**\n\n{answer}")
                    
                    # Clear the selected question
                    st.session_state.selected_question = None
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.selected_question = None

if __name__ == "__main__":
    main()
