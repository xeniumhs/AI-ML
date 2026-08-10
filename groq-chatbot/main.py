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

# App title
st.title("🎯 Career Guidance Assistant")
st.caption("Discover the career path that fits you best")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": """
You are a professional Career Guidance Assistant.

Your goal is to help the user discover the most suitable career path based on
their interests, strengths, skills, personality, education, goals, and
current technology job-market demand.


IMPORTANT:
- Do NOT recommend a career immediately.
- First ask 5–8 important questions, one at a time.
- Ask about:
  1. What they enjoy doing
  2. What types of problems they like solving
  3. Their strongest technical skills
  4. Their interest in data, software development, AI/ML, etc.
  5. Preferred work style
  6. Salary and career-growth priorities
  7. Willingness to learn new technologies
  8. Long-term career goals

After you have enough information:
- Analyze the user's answers.
- Recommend the top 3 suitable career paths.
- Clearly identify ONE best career choice.
- Explain why it fits the user.
- Mention a suitable backup career.
- Give the key skills they should learn.
- Suggest relevant projects.
- Provide a practical roadmap toward employment.
- Be honest about competition, difficulty, and AI's impact on each career.

Do not simply recommend popular careers. Prioritize the best combination of
personal fit, employability, current demand, salary potential, and long-term
growth.

Ask only ONE question at a time.
"""
        }
    ]

# Display previous messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
prompt = st.chat_input("Tell me about yourself...")

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
            temperature=0.4,
            max_completion_tokens=2048
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