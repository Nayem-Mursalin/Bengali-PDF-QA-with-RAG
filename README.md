
# Bengali PDF Question Answering with RAG 🇧🇩

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about Bengali PDF documents and receive context-aware answers through a conversational interface.

The system combines document processing, semantic search, vector storage, and Large Language Models (LLMs) to make information in Bengali documents easier to access.

## 🚀 Overview

Finding specific information in a long PDF can be time-consuming, especially when the document is written in Bengali.

This project addresses that problem by building a question-answering system that retrieves relevant information from a PDF and uses an LLM to generate answers grounded in the document's content.

Instead of relying only on the model's pretrained knowledge, the system retrieves relevant document chunks and provides them as context for answer generation.

## ✨ Key Features

- **Bengali PDF Question Answering:** Ask natural-language questions about document content.
- **Retrieval-Augmented Generation:** Combine semantic retrieval with LLM-based answer generation.
- **Semantic Search:** Retrieve relevant text chunks using sentence embeddings.
- **Vector Database:** Store and search document embeddings using ChromaDB.
- **Source-Aware Responses:** Provide source-page references to help users verify answers.
- **Interactive Chat Interface:** Explore document content through a Gradio interface.
- **LLM Integration:** Use the Groq API for language-model inference.

## 🧠 System Architecture

The application follows a standard Retrieval-Augmented Generation workflow.

### 1. Document Ingestion

- Load and extract text from the input PDF.
- Split the extracted text into smaller, manageable chunks.
- Preserve relevant document metadata, such as page references.

### 2. Embedding Generation and Indexing

- Convert text chunks into numerical vector embeddings using Sentence Transformers.
- Store embeddings and associated document metadata in ChromaDB.
- Prepare the vector store for semantic retrieval.

### 3. Question Answering

- Accept a user's question through the chat interface.
- Convert the question into an embedding.
- Retrieve relevant document chunks from the vector database.
- Pass the retrieved context and question to the LLM.
- Generate a response grounded in the retrieved document content.

### Workflow

```text
                 Bengali PDF
                      |
                      v
               Text Extraction
                      |
                      v
                Text Chunking
                      |
                      v
             Embedding Generation
                      |
                      v
                  ChromaDB
                      |
                      v
User Question --> Semantic Retrieval
                      |
                      v
            Relevant Document Chunks
                      |
                      v
                 Groq LLM API
                      |
                      v
              Generated Answer
                      |
                      v
               Gradio Interface
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| RAG Framework | LangChain |
| Language Model Inference | Groq API |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| User Interface | Gradio |
| Development Environment | Jupyter Notebook / Python |

## 📂 Project Structure

```text
Bengali-PDF-QA-with-RAG/
├── README.md
├── requirements.txt
├── data/                  # Input PDFs, if included
├── notebooks/             # Notebooks, if included
├── src/                   # Application modules, if included
└── ...                    # Other project files
```

> Update this structure to match the actual files and directories in the repository.

## ⚙️ Installation and Setup

### Prerequisites

- Python
- Git
- A Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/Nayem-Mursalin/Bengali-PDF-QA-with-RAG.git

cd Bengali-PDF-QA-with-RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API Key

If the application uses environment variables, create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your API key from the [Groq Console](https://console.groq.com/).

**Security:** Never commit your `.env` file or expose your API key in source code.

### 5. Run the Application

Run the actual application entry point. For example, if the main script is `app.py`:

```bash
python app.py
```

Replace `app.py` with the correct filename if your project uses a different entry point.

## 💬 Example Usage

After launching the application:

1. Open the Gradio interface.
2. Load or select the supported Bengali PDF, if required.
3. Enter a question about the document.
4. Review the generated answer and any available source-page references.

### Example Questions

**In Bengali**

- এই ডকুমেন্টের মূল বিষয় কী?
- ডকুমেন্টে উল্লেখ করা প্রধান বিষয়গুলো কী কী?
- নির্দিষ্ট কোনো তথ্য কোন পৃষ্ঠায় উল্লেখ করা হয়েছে?

**In English**

- What are the main points discussed in this document?
- What does the document say about a specific topic?
- On which page can I find a particular piece of information?

The generated answers depend on the content of the PDF provided to the system.

## 🔍 Why RAG?

Traditional LLM-based question answering can produce responses that are not supported by a specific document.

RAG improves document-grounded question answering by retrieving relevant information before generating a response.

### Benefits

- Access to information contained in the supplied document.
- Reduced reliance on the model's pretrained knowledge alone.
- Better traceability when source references are available.
- A practical way to build question-answering applications without training an LLM from scratch.

RAG does not guarantee factual accuracy. PDF extraction, chunking, embedding quality, retrieval relevance, and LLM behavior all affect the final answer.

## 🔮 Future Improvements

- Support multiple PDF uploads and document collections.
- Improve Bengali text extraction and preprocessing.
- Experiment with Bengali-specific and multilingual embedding models.
- Add hybrid search and reranking to improve retrieval relevance.
- Implement automated RAG evaluation using retrieval and answer-quality metrics.
- Add conversation history and document-level filtering.
- Deploy the application with a production-ready interface and backend.

## 👨‍💻 Author

**MD. Sayed Al Nayem**

- LinkedIn: [sayedalnayem](https://www.linkedin.com/in/sayedalnayem/)

---

*Built to explore practical applications of Retrieval-Augmented Generation, semantic search, and Bengali-language AI.*
