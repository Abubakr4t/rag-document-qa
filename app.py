import os
import uuid
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ============================================================
# 👨‍💻 DEVELOPER CONFIG - edit these
# ============================================================
DEVELOPER_1_NAME = "Muhammad Abubakar"
DEVELOPER_1_ROLE = "ML Engineer"
DEVELOPER_1_LINKEDIN = "https://www.linkedin.com/in/abubakr4t"
DEVELOPER_1_INITIALS = "MA"

DEVELOPER_2_NAME = "Muhammad Zeeshan Asif"
DEVELOPER_2_ROLE = "ML Engineer"
DEVELOPER_2_LINKEDIN = "https://www.linkedin.com/in/zeeshan-asif-75bb57312"
DEVELOPER_2_INITIALS = "MZ"

#PROJECT_GITHUB = "https://github.com/abubakr4t/rag-document-qa"  # update if different

# ============================================================
# 🎨 THEME - Professional indigo/violet
# ============================================================
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
)

# ============================================================
# Backend setup (unchanged)
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("Embedding model loaded.")

prompt_template = """You are a knowledgeable assistant helping users understand a document. Use the provided context to answer questions thoroughly and accurately.

Guidelines:
- Synthesize information from multiple parts of the context to give complete answers
- If the context contains relevant information, USE IT — don't say "no information" when partial info exists
- Cite specific page numbers when referencing facts
- Only say "I couldn't find this information in the document" if the context is completely unrelated to the question
- Be informative and helpful, not overly cautious

Context from the document:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

current_retriever = {"retriever": None, "filename": None}


def process_pdf(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf_{uuid.uuid4().hex[:8]}"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return retriever, len(documents), len(chunks)


def upload_pdf(pdf_file):
    if pdf_file is None:
        return """<div class="status-card status-error">
            <div class="status-icon">⚠️</div>
            <div class="status-text">Please upload a PDF file first</div>
        </div>"""

    try:
        retriever, num_pages, num_chunks = process_pdf(pdf_file)
        current_retriever["retriever"] = retriever
        current_retriever["filename"] = pdf_file.split("/")[-1]
        return f"""<div class="status-card status-success">
            <div class="status-header">
                <span class="status-icon">✓</span>
                <span class="status-title">Document Ready</span>
            </div>
            <div class="status-filename">{current_retriever['filename']}</div>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{num_pages}</div>
                    <div class="stat-label">Pages</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{num_chunks}</div>
                    <div class="stat-label">Chunks</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">384</div>
                    <div class="stat-label">Dimensions</div>
                </div>
            </div>
            <div class="status-hint">Ask questions about this document below →</div>
        </div>"""
    except Exception as e:
        return f"""<div class="status-card status-error">
            <div class="status-icon">⚠️</div>
            <div class="status-text">Error: {str(e)}</div>
        </div>"""


def chat_with_pdf(message, history):
    if current_retriever["retriever"] is None:
        return "⚠️ **Please upload a PDF first** before asking questions."

    try:
        retrieved_docs = current_retriever["retriever"].invoke(message)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        formatted_prompt = PROMPT.format(context=context, question=message)
        response = llm.invoke(formatted_prompt)
        answer = response.content

        seen = set()
        unique_docs = []
        for doc in retrieved_docs:
            key = (doc.metadata.get('page'), doc.page_content[:50])
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        sources_text = "\n\n---\n### 📚 Sources\n"
        for i, doc in enumerate(unique_docs, 1):
            page = doc.metadata.get('page', 'N/A')
            preview = doc.page_content[:120].replace('\n', ' ').strip()
            sources_text += f"\n**`[{i}]`** Page **{page}** — _{preview}..._\n"

        return answer + sources_text

    except Exception as e:
        return f"❌ **Error:** {str(e)}"


# ============================================================
# 🎨 PROFESSIONAL CSS
# ============================================================
custom_css = """
* { 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto;
}

/* ============ HERO HEADER ============ */
.hero-section {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
    background-size: 200% 200%;
    animation: gradientShift 15s ease infinite;
    border-radius: 20px;
    padding: 48px 32px;
    margin-bottom: 24px;
    text-align: center;
    color: white;
    box-shadow: 0 20px 60px rgba(79, 70, 229, 0.3);
    position: relative;
    overflow: hidden;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.85em;
    font-weight: 500;
    margin-bottom: 16px;
    letter-spacing: 0.5px;
}

.hero-title {
    font-size: 3em !important;
    font-weight: 800 !important;
    margin: 0 0 12px 0 !important;
    color: white !important;
    letter-spacing: -1px;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.15em !important;
    margin: 0 auto 24px auto !important;
    color: rgba(255, 255, 255, 0.92) !important;
    max-width: 700px;
    line-height: 1.5;
}

.hero-stack {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 24px;
}

.stack-pill {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.85em;
    font-weight: 500;
}

/* ============ SECTION HEADINGS ============ */
.section-heading {
    font-size: 1.1em;
    font-weight: 700;
    color: var(--body-text-color);
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 0.85em;
    font-weight: 700;
}

/* ============ STATUS CARD ============ */
.status-card {
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    border: 1px solid rgba(99, 102, 241, 0.2);
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.05), rgba(124, 58, 237, 0.05));
}

.status-success {
    border-color: rgba(34, 197, 94, 0.3);
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.08), rgba(34, 197, 94, 0.03));
}

.status-error {
    border-color: rgba(239, 68, 68, 0.3);
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.03));
}

.status-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}

.status-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #22c55e;
    color: white;
    font-weight: 700;
    font-size: 0.9em;
}

.status-error .status-icon {
    background: #ef4444;
}

.status-title {
    font-weight: 700;
    font-size: 1.05em;
    color: var(--body-text-color);
}

.status-filename {
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 0.85em;
    background: rgba(99, 102, 241, 0.1);
    padding: 6px 10px;
    border-radius: 6px;
    display: inline-block;
    margin: 8px 0;
    color: var(--body-text-color);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 12px 0;
}

.stat-item {
    background: rgba(99, 102, 241, 0.08);
    border-radius: 8px;
    padding: 10px;
    text-align: center;
}

.stat-value {
    font-size: 1.5em;
    font-weight: 800;
    color: #6366f1;
    line-height: 1;
}

.stat-label {
    font-size: 0.75em;
    color: var(--body-text-color-subdued);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

.status-hint {
    font-size: 0.85em;
    color: var(--body-text-color-subdued);
    margin-top: 8px;
    font-style: italic;
}

.status-text {
    color: var(--body-text-color);
    font-weight: 500;
}

/* ============ INFO BOX ============ */
.info-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(168, 85, 247, 0.05));
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 16px;
    margin-top: 16px;
}

.info-box-title {
    font-weight: 700;
    color: var(--body-text-color);
    margin-bottom: 8px;
    font-size: 0.95em;
}

.info-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 6px 0;
    font-size: 0.9em;
    color: var(--body-text-color-subdued);
}

.step-num {
    background: #6366f1;
    color: white;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75em;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
}

/* ============ FOOTER ============ */
.footer-section {
    margin-top: 40px;
    padding: 32px 24px;
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.04), rgba(124, 58, 237, 0.04));
    border-radius: 16px;
    border: 1px solid rgba(99, 102, 241, 0.15);
}

.footer-title {
    text-align: center;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--body-text-color-subdued);
    margin-bottom: 20px;
    font-weight: 600;
}

.dev-grid {
    display: flex;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 24px;
}

.dev-card {
    background: var(--background-fill-primary);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 14px;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: all 0.3s ease;
    min-width: 280px;
    text-decoration: none;
}

.dev-card:hover {
    border-color: #6366f1;
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.15);
}

.dev-avatar {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 1.1em;
    flex-shrink: 0;
}

.dev-info {
    flex: 1;
    text-align: left;
}

.dev-name {
    font-weight: 700;
    color: var(--body-text-color);
    font-size: 0.95em;
    margin: 0;
}

.dev-role {
    font-size: 0.8em;
    color: var(--body-text-color-subdued);
    margin: 2px 0 0 0;
}

.dev-link {
    color: #6366f1;
    font-size: 1.3em;
    text-decoration: none;
    transition: transform 0.2s;
}

.dev-link:hover {
    transform: scale(1.2);
}

.footer-stack {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 6px;
    margin: 20px 0 16px 0;
}

.footer-stack .stack-pill {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    color: var(--body-text-color);
    font-size: 0.75em;
}

.footer-bottom {
    text-align: center;
    font-size: 0.8em;
    color: var(--body-text-color-subdued);
    padding-top: 16px;
    border-top: 1px solid rgba(99, 102, 241, 0.1);
    margin-top: 16px;
}

.footer-bottom a {
    color: #6366f1;
    text-decoration: none;
    font-weight: 500;
}

.footer-bottom a:hover {
    text-decoration: underline;
}
"""

# ============================================================
# 🏗️ BUILD UI
# ============================================================
with gr.Blocks(theme=THEME, css=custom_css, title="DocuChat AI") as demo:

    # HERO
    gr.HTML("""
    <div class="hero-section">
        <div class="hero-badge">✨ AI-Powered Document Intelligence</div>
        <h1 class="hero-title">DocuChat AI</h1>
        <p class="hero-subtitle">
            Upload any PDF and ask questions in natural language.
            Get accurate answers with source citations — powered by Retrieval-Augmented Generation.
        </p>
        <div class="hero-stack">
            <span class="stack-pill">⚡ Llama 3.3 70B</span>
            <span class="stack-pill">🔍 ChromaDB</span>
            <span class="stack-pill">🧠 LangChain</span>
            <span class="stack-pill">⚙️ Groq</span>
        </div>
    </div>
    """)

    with gr.Row():
        # LEFT: Upload
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="section-heading">
                <span class="section-number">1</span>
                <span>Upload your document</span>
            </div>
            """)

            pdf_input = gr.File(
                label="",
                file_types=[".pdf"],
                type="filepath"
            )

            upload_btn = gr.Button("⚡ Process Document", variant="primary", size="lg")

            upload_status = gr.HTML(
                """<div class="status-card">
                    <div class="status-text" style="text-align: center; color: var(--body-text-color-subdued);">
                        📄 No document uploaded yet
                    </div>
                </div>"""
            )

            gr.HTML("""
            <div class="info-box">
                <div class="info-box-title">💡 How it works</div>
                <div class="info-step"><span class="step-num">1</span>Upload your PDF document</div>
                <div class="info-step"><span class="step-num">2</span>We split & embed the content</div>
                <div class="info-step"><span class="step-num">3</span>Ask questions in plain English</div>
                <div class="info-step"><span class="step-num">4</span>Get answers with citations</div>
            </div>
            """)

        # RIGHT: Chat
        with gr.Column(scale=2):
            gr.HTML("""
            <div class="section-heading">
                <span class="section-number">2</span>
                <span>Ask questions</span>
            </div>
            """)

            chatbot = gr.ChatInterface(
                fn=chat_with_pdf,
                examples=[
                    "What is this document about?",
                    "Summarize the key points",
                    "What are the main contributions?",
                    "What datasets were used?",
                    "Make study notes from this document",
                ],
                cache_examples=False
            )

    upload_btn.click(fn=upload_pdf, inputs=pdf_input, outputs=upload_status)

    # FOOTER
    gr.HTML(f"""
    <div class="footer-section">
        <div class="footer-title">Built By</div>
        <div class="dev-grid">
            <a href="{DEVELOPER_1_LINKEDIN}" target="_blank" class="dev-card">
                <div class="dev-avatar">{DEVELOPER_1_INITIALS}</div>
                <div class="dev-info">
                    <p class="dev-name">{DEVELOPER_1_NAME}</p>
                    <p class="dev-role">{DEVELOPER_1_ROLE}</p>
                </div>
                <span class="dev-link">→</span>
            </a>
            <a href="{DEVELOPER_2_LINKEDIN}" target="_blank" class="dev-card">
                <div class="dev-avatar">{DEVELOPER_2_INITIALS}</div>
                <div class="dev-info">
                    <p class="dev-name">{DEVELOPER_2_NAME}</p>
                    <p class="dev-role">{DEVELOPER_2_ROLE}</p>
                </div>
                <span class="dev-link">→</span>
            </a>
        </div>
        <div class="footer-stack">
            <span class="stack-pill">Python 3.11</span>
            <span class="stack-pill">LangChain</span>
            <span class="stack-pill">ChromaDB</span>
            <span class="stack-pill">Llama 3.3 70B</span>
            <span class="stack-pill">Sentence Transformers</span>
            <span class="stack-pill">Gradio</span>
        </div>
        <div class="footer-bottom">
            <a href="{PROJECT_GITHUB}" target="_blank">⭐ Star on GitHub</a>
            &nbsp; · &nbsp;
            <span>© 2026 · RAG-based Document Q&A System</span>
        </div>
    </div>
    """)


if __name__ == "__main__":
    demo.launch()