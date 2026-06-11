import os
from dotenv import load_dotenv
import gradio as gr

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

prompt_template = """তুমি একজন AI। নিচের context দিয়ে প্রশ্নের উত্তর দাও।
Context থেকে উত্তর না পেলে বলো "এই তথ্য ডকুমেন্টে নেই।"

Context: {context}
প্রশ্ন: {input}
উত্তর:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

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

    def chain(question):
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt_text = PROMPT.format(context=context, input=question)
        response = llm.invoke(prompt_text)
        return response.content, docs

    return chain


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
        answer, sources = qa_chain(question)
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


# UI
with gr.Blocks(title="RAG App") as demo:
    gr.Markdown("## PDF Question Answering")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="PDF Upload", file_types=[".pdf"])
            load_btn = gr.Button("Load করো", variant="primary")
            status_box = gr.Textbox(label="Status", lines=3, interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=400)
            question_input = gr.Textbox(label="প্রশ্ন করো")
            with gr.Row():
                ask_btn = gr.Button("জিজ্ঞেস করো", variant="primary")
                clear_btn = gr.Button("Clear")

    load_btn.click(fn=load_pdf, inputs=pdf_input, outputs=status_box)
    ask_btn.click(fn=answer_question, inputs=[question_input, chatbot], outputs=chatbot)
    question_input.submit(
        fn=answer_question, inputs=[question_input, chatbot], outputs=chatbot
    )
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, question_input])

demo.launch()
