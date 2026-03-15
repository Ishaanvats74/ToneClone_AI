import re
import os
import random
import hashlib
from fastapi import FastAPI, UploadFile, Form, File
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChatState(TypedDict):
    chat_history: list
    user_name: str
    namespace: str      
    query: str
    file_path: str
    ai_reply: str

load_dotenv()
app = FastAPI()


# ── Clients ────────────────────────────────────────────────────────────────────
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview",temperature=0.7,api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "toneclone"
pc.delete_index("toneclone")

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(INDEX_NAME)

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_vector_store(namespace: str) -> PineconeVectorStore:
    """Return a namespace-scoped vector store for this user session."""
    return PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        namespace=namespace,        # ← each user is fully isolated here
    )

def make_namespace(user_name: str, filename: str) -> str:
    """
    Stable, unique namespace per (user, chat-file) pair.
    Hash keeps it short and URL-safe.
    """
    raw = f"{user_name.strip().lower()}::{filename}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]

# ── Graph nodes ────────────────────────────────────────────────────────────────

def load_format(state: ChatState):
    messages = []
    with open(state["file_path"], encoding="utf-8") as f:
        for line in f:
            pattern = r"\d{2}/\d{2}/\d{2}, .* - (.*?): (.*)"
            match = re.match(pattern, line)
            if not match:
                continue
            sender, message = match.group(1), match.group(2)
            if not message.strip():
                continue
            if "disappearing messages" in message.lower():
                continue
            if "<Media omitted>" in message:
                continue
            if state["user_name"] == sender:
                continue
            messages.append(message.strip())
    return {"chat_history": messages}


def chunk_texts(state: ChatState):
    text = "\n".join(state["chat_history"])
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)

    vs = get_vector_store(state["namespace"])
    vs.add_texts(chunks)                        # stored under user's namespace
    return {}


def generate_response(state: ChatState):
    vs = get_vector_store(state["namespace"])    # reads only this user's data
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    semantic_docs = retriever.invoke(state["query"])
    semantic_context = "\n".join([d.page_content for d in semantic_docs])

    # Fetch random style examples from this user's namespace only
    raw = pinecone_index.query(
        vector=[0.0] * 1024,           # dummy vector — we only want metadata/text
        top_k=30,
        namespace=state["namespace"],
        include_metadata=True,
    )
    all_examples = [m["metadata"].get("text", "") for m in raw["matches"]]
    style_examples = "\n".join(random.sample(all_examples, min(8, len(all_examples))))

    context = semantic_context + "\n" + style_examples

    template = PromptTemplate(
        template="""
            You are mimicking someone's WhatsApp texting style.

            STYLE EXAMPLES
            --------------
            {context}

            These are ONLY examples of writing style.
            Do NOT reuse the content of these messages.
            Use them only to understand: tone, wording, message length, slang.

            Rules:
            - Reply must be a complete natural message.
            - Use the SAME texting style.
            - NEVER copy any sentence from the examples.
            - The reply must be completely new.
            - Reply must be 1–6 words.
            - Casual Hinglish WhatsApp style.
            - Do NOT include any names.

            Message from {user_name}:
            {query}

            Reply as the other person would.
            Output ONLY the reply.
        """,
        input_variables=["query", "context", "user_name"],
    )

    prompt = template.format(
        context=context,
        query=state["query"],
        user_name=state["user_name"],
    )
    response = llm.invoke(prompt)
    return {"ai_reply": response.content}


# ── Graphs ─────────────────────────────────────────────────────────────────────

index_graph = StateGraph(ChatState)
index_graph.add_node("load_format", load_format)
index_graph.add_node("chunk_texts", chunk_texts)
index_graph.add_edge(START, "load_format")
index_graph.add_edge("load_format", "chunk_texts")
index_graph.add_edge("chunk_texts", END)
index_graph = index_graph.compile()

chat_graph = StateGraph(ChatState)
chat_graph.add_node("generate_response", generate_response)
chat_graph.add_edge(START, "generate_response")
chat_graph.add_edge("generate_response", END)
chat_graph = chat_graph.compile()

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/load_chat")
def load_chat(file: UploadFile = File(...), user_name: str = Form(...)):
    namespace = make_namespace(user_name, file.filename)
    file_path = file.filename

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    index_graph.invoke({
        "file_path": file_path,
        "user_name": user_name,
        "namespace": namespace,
    })

    # Return namespace so frontend can store it in session
    return {"status": "chat indexed", "namespace": namespace}


@app.post("/chat")
def chat(query: str, namespace: str):          
    result = chat_graph.invoke({
        "query": query,
        "user_name": "",                       
        "namespace": namespace,
    })

    return {"reply": result["ai_reply"][0]['text']}