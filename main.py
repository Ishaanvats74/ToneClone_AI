import re
import os
import random
from fastapi import FastAPI , UploadFile
from dotenv import load_dotenv
from langchain_chroma import Chroma
from typing import TypedDict , Literal
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph , START , END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings

class ChatState(TypedDict):
    chat_history: list
    user_name: str
    query: str    
    file_path:str
    ai_reply:str
    text_chuked: Literal["yes", "no"]

load_dotenv()
app = FastAPI()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview",temperature=0.7,api_key=os.getenv("GOOGLE_API_KEY"))
# embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
# llm = ChatOllama(model="qwen2.5:7b",temperature=0.7)
vector_store = Chroma(collection_name="chats",embedding_function=embeddings)
current_user_name = None

@app.get("/")
def read_root():
    return {"Hello": "World"}





def load_format(state:ChatState):
    messages = []
    
    with open(state["file_path"], encoding="utf-8") as f:
        for line in f:

            pattern = r"\d{2}/\d{2}/\d{2}, .* - (.*?): (.*)"
            match = re.match(pattern, line)

            if match :
                sender = match.group(1)
                message = match.group(2)

                if message.strip() == "":
                    continue

                if "disappearing messages" in message.lower():
                    continue
                if "<Media omitted>" in message:
                    continue
                if state['user_name'] == sender:
                    continue
                message = message.strip()
                messages.append(message)

    return {"chat_history":messages}



def chunk_texts(state: ChatState):
    text = "\n".join(state["chat_history"])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    vector_store.add_texts(texts=chunks)
    return {"text_chuked": "yes"}
    
def check_chunk(state: ChatState):

    if state["text_chuked"] == "yes":
        return "generate_response"

    return "chunk_texts"

def generate_response(state: ChatState):
    user_name = state["user_name"]
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    semantic_docs = retriever.invoke(state["query"])

    semantic_context = "\n".join([doc.page_content for doc in semantic_docs]) 

    style_docs = vector_store._collection.get(limit=30)
    examples = style_docs["documents"]

    style_examples = "\n".join(random.sample(examples, 8))

    context = semantic_context + "\n" + style_examples
    template = PromptTemplate(
       template = """
            You are mimicking someone's WhatsApp texting style.

            STYLE EXAMPLES
            --------------
            {context}

            These are ONLY examples of writing style.

            Do NOT reuse the content of these messages.
            Use them only to understand:
            - tone
            - wording
            - message length
            - slang

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
        input_variables=["query", "context","user_name"]
    )

    prompt = template.format(
        context=context,
        query=state["query"],
        user_name=user_name
    )
    response = llm.invoke(prompt)
    return {"ai_reply":response.content}


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

@app.post("/load_chat")
def load_chat(file: UploadFile, user_name: str):

    global current_user_name
    current_user_name = user_name
    file_path = f"{file.filename}"

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    index_graph.invoke({
        "file_path": file_path,
        "user_name": user_name,
        "text_chuked": "no"
    })

    return {"status": "chat indexed"}

@app.post("/chat")
def chat(query: str):

    result = chat_graph.invoke({
        "query": query,
        "user_name": current_user_name
    })

    return {"reply": result["ai_reply"]}
