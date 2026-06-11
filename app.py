import os
from dotenv import load_dotenv
import gradio as gr

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1000,
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# prompt_template = """You are an AI assistant. Answer the question based on the context below.
prompt_template = """তুমি একজন AI। নিচের context দিয়ে প্রশ্নের উত্তর দাও।
Context থেকে উত্তর না পেলে বলো "এই তথ্য ডকুমেন্টে নেই।"

Context: {context}
প্রশ্ন: {question}
উত্তর:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = None


def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return splitter.split_documents(docs)


def create_vectorstore(chunks):
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory="./chroma_db"
    )
    return vectorstore


def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return qa_chain


def load_pdf(pdf_file):
    global qa_chain
    if pdf_file is None:
        return "কোনো PDF select করোনি।"
    try:
        chunks = load_and_split_pdf(pdf_file.name)
        vectorstore = create_vectorstore(chunks)
        qa_chain = create_qa_chain(vectorstore)
        return f"সফল! Chunks: {len(chunks)} — এখন প্রশ্ন করো।"
    except Exception as e:
        return f"Error: {str(e)}"


def answer_question(question, history):
    global qa_chain
    if qa_chain is None:
        return history + [
            gr.ChatMessage(role="user", content=question),
            gr.ChatMessage(role="assistant", content="আগে PDF upload করো।"),
        ]
    if not question.strip():
        return history
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        sources = result.get("source_documents", [])
        if sources:
            pages = sorted(set(doc.metadata.get("page", 0) + 1 for doc in sources))
            answer += f"\n\n Pages: {pages}"
        return history + [
            gr.ChatMessage(role="user", content=question),
            gr.ChatMessage(role="assistant", content=answer),
        ]
    except Exception as e:
        return history + [
            gr.ChatMessage(role="user", content=question),
            gr.ChatMessage(role="assistant", content=f"Error: {str(e)}"),
        ]
