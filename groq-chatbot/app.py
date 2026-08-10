import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()

# Initialize Groq
client = Groq(api_key=api_key)

st.title("🤖 Groq Chatbot")
st.caption("Powered by Groq API")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        }
    ]

# Display previous messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask something...")

if prompt:

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Send request to Groq
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=st.session_state.messages,
            temperature=0.7,
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        # Display response
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Save response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:
        st.error(f"Error: {e}")