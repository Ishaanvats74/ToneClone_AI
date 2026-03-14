import streamlit as st
import requests

API_URL = "https://toneclone-ai.onrender.com"

st.set_page_config(page_title="Tone Clone AI", layout="centered")

st.title("📱 WhatsApp Tone Clone AI")

# -----------------------------
# Session State Initialization
# -----------------------------
if "chat_loaded" not in st.session_state:
    st.session_state.chat_loaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# =============================
# Upload Chat Section
# =============================

st.header("1️⃣ Upload WhatsApp Chat")

user_name = st.text_input("Your Name in the Chat")

uploaded_file = st.file_uploader(
    "Upload exported WhatsApp chat (.txt)",
    type=["txt"]
)

if st.button("Load Chat"):

    if uploaded_file and user_name:

        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue())
        }

        params = {
            "user_name": user_name
        }

        with st.spinner("Indexing chat... ⏳"):

            response = requests.post(
                f"{API_URL}/load_chat",
                files=files,
                params=params
            )

        if response.status_code == 200:
            st.success("Chat indexed successfully! ✅")
            st.session_state.chat_loaded = True
        else:
            st.error("Error indexing chat ❌")

    else:
        st.warning("Please upload a chat file and enter your name.")


# =============================
# Chat Section
# =============================

if st.session_state.chat_loaded:

    st.header("2️⃣ Chat")

    # Display previous chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Type your message...")

    if user_input:

        # Store user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Assistant reply section
        with st.chat_message("assistant"):

            with st.spinner("Typing... ⌛"):

                response = requests.post(
                    f"{API_URL}/chat",
                    params={
                        "query": user_input,
                        "user_name": user_name
                    }
                )

                if response.status_code == 200:

                    reply = response.json()["reply"]

                    # Save assistant message
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )

                    st.write(reply)

                else:
                    st.error("Error getting reply ❌")

