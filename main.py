import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from typing import TypedDict , Literal
from langgraph.graph import StateGraph , START , END
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
import os
import random
from dotenv import load_dotenv
load_dotenv()

# embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
# llm = ChatOllama(model="qwen2.5:7b",temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview",temperature=0.7,api_key=os.getenv("GOOGLE_API_KEY"))
vector_store = Chroma(collection_name="chats",embedding_function=embeddings)

class ChatState(TypedDict):
    chat_history: list
    user_name: str
    query: str    
    file_path:str
    ai_reply:str
    text_chuked: Literal["yes", "no"]

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

    semantic_context = "\n".join(
        [doc.page_content for doc in semantic_docs]
    )

    # random style examples
    

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


graph = StateGraph(ChatState)

graph.add_node("load_format", load_format)
graph.add_node("chunk_texts", chunk_texts)
graph.add_node("generate_response", generate_response)

graph.add_edge(START, "load_format")
graph.add_conditional_edges("load_format",check_chunk,{"generate_response": "generate_response", "chunk_texts": "chunk_texts"})
graph.add_edge("chunk_texts", "generate_response")
graph.add_edge("generate_response", END)

app = graph.compile()

result = app.invoke({
    "file_path": "chat.txt",
    "query": "kya kar rahe ho ?",
    "user_name": "Ishaan Vats",
    "text_chuked": "no"
})
print(result["ai_reply"][0]['text'])   



