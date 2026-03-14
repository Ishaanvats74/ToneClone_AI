import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from typing import TypedDict , Literal
from langgraph.graph import StateGraph , START , END

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
llm = ChatOllama(model="llama3.1:8b",temperature=0)
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

                messages.append(f"{sender}: {message}")

    return {"chat_history":messages}



def chunk_texts(state: ChatState):
    text = "\n".join(state["chat_history"])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=80,chunk_overlap=20)
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
    style_docs = vector_store._collection.get(limit=4)

    style_examples = "\n".join(
        doc for sublist in style_docs["documents"] for doc in sublist
    )

    context = semantic_context + "\n" + style_examples

    template = PromptTemplate(
        template = """
You are continuing a WhatsApp chat.

STYLE EXAMPLES (how the other person writes)
---------------------------------------------
{context}

These examples show ONLY the person's texting style.
If a message in the examples answers a different question,
do NOT reuse it.

Rules:
- Use the SAME style as the examples.
- Replies must be SHORT (1–8 words).
- Use casual Hinglish like WhatsApp messages.
- Do NOT write formal Hindi.
- Do NOT copy the example messages.
- If examples answer different questions, ignore them.
- Generate a NEW reply.
- Do NOT include the speaker name.

MESSAGE FROM {user_name}:
{query}

Reply like the other person would on WhatsApp.
Only output the message.
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
    "query": "bhaiya kaha ho?",
    "user_name": "Ishaan Vats",
    "text_chuked": "no"
})
print(result["ai_reply"])   



