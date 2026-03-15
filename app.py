import streamlit as st
import requests

# API_URL = "https://toneclone-ai.onrender.com"
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Tone Clone AI", layout="centered")
st.title("📱 WhatsApp Tone Clone AI")

if "chat_loaded" not in st.session_state:
    st.session_state.chat_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "namespace" not in st.session_state:   # ← store namespace per browser tab
    st.session_state.namespace = None

st.header("1️⃣ Upload WhatsApp Chat")
user_name = st.text_input("Your Name in the Chat")
uploaded_file = st.file_uploader("Upload exported WhatsApp chat (.txt)", type=["txt"])

if st.button("Load Chat"):
    if uploaded_file and user_name:
        with st.spinner("Indexing chat... ⏳"):
            response = requests.post(
                f"{API_URL}/load_chat",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                data={"user_name": user_name},
            )
        if response.status_code == 200:
            data = response.json()
            st.session_state.namespace = data["namespace"]   # ← save it
            st.session_state.chat_loaded = True
            st.success("Chat indexed successfully! ✅")
        else:
            st.error(response.text)
    else:
        st.warning("Please upload a chat file and enter your name.")

if st.session_state.chat_loaded:
    st.header("2️⃣ Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Typing... ⌛"):
                response = requests.post(
                    f"{API_URL}/chat",
                    params={
                        "query": user_input,
                        "namespace": st.session_state.namespace,   # ← pass it
                    },
                )
                if response.status_code == 200:
                    reply = response.json()["reply"]
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.write(reply)
                else:
                    st.error("Error getting reply ❌")
