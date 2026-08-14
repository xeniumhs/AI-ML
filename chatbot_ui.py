import streamlit as st
import ollama
import psycopg2
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Interviewer",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

@st.cache_resource
def get_database_connection():

    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            port=5433,
            database="ragdb",
            user="postgres",
            password="postgres"
        )

        return conn

    except psycopg2.Error as e:

        st.error("POSTGRESQL ERROR:")
        st.error(str(e))

        return None


conn = get_database_connection()


if conn is None:
    st.stop()


cur = conn.cursor()


# ============================================================
# CREATE TABLE
# ============================================================

cur.execute("""
CREATE TABLE IF NOT EXISTS interview_questions (
    id BIGSERIAL PRIMARY KEY,
    question TEXT,
    category TEXT,
    role TEXT,
    experience TEXT,
    difficulty TEXT,
    source_type TEXT,
    ideal_answer TEXT,
    keywords TEXT
);
""")

conn.commit()


# ============================================================
# LOAD NOMIC EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True
    )

    return model


model = load_embedding_model()


# ============================================================
# 1. SEARCH QUESTIONS USING NOMIC + PGVECTOR
# ============================================================

def search_questions(query, limit=5):

    query_embedding = model.encode(query).tolist()

    cur.execute("""
        SELECT
            id,
            question,
            category,
            role,
            experience,
            difficulty,
            ideal_answer,
            keywords,
            1 - (embedding <=> %s::vector) AS similarity
        FROM interview_questions
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (
        str(query_embedding),
        str(query_embedding),
        limit
    ))

    results = cur.fetchall()

    return results


# ============================================================
# 2. BUILD CONTEXT
# ============================================================

def build_context(results):

    return "\n\n".join([
        f"""
Question: {row[1]}
Category: {row[2]}
Role: {row[3]}
Experience: {row[4]}
Difficulty: {row[5]}
Ideal Answer: {row[6]}
Keywords: {row[7]}
Similarity: {row[8]:.3f}
"""
        for row in results
    ])


# ============================================================
# 3. GENERATE INTERVIEW QUESTION
# ============================================================

def generate_question(
    role,
    previous_answer=None,
    evaluation=None,
    asked_questions=None
):

    if asked_questions is None:
        asked_questions = []

    # Search based on role for first question
    if previous_answer is None:

        search_query = role

    # Search based on previous answer for adaptive question
    else:

        search_query = f"""
        Role: {role}

        Candidate previous answer:
        {previous_answer}

        Previous evaluation:
        {evaluation}
        """

    results = search_questions(
        search_query,
        limit=5
    )

    context = build_context(results)

    already_asked = "\n".join(
        f"- {q}" for q in asked_questions
    )

    prompt = f"""
You are a professional AI interviewer.

Candidate role:
{role}

Interview knowledge retrieved from PostgreSQL:
{context}

Questions already asked:
{already_asked}

Previous candidate answer:
{previous_answer}

Previous evaluation:
{evaluation}

Your task:

Generate ONE interview question.

Rules:
- Ask exactly ONE question.
- Make it relevant to the candidate's role.
- Do not repeat previous questions.
- Do not give the answer.
- If the previous answer was weak, ask a useful follow-up or test that weak area.
- If the previous answer was strong, increase the difficulty slightly.
- Keep the question realistic for a job interview.
- Return ONLY the question.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


# ============================================================
# 4. EVALUATE INDIVIDUAL ANSWER
# ============================================================

def evaluate_answer(
    role,
    question,
    answer
):

    results = search_questions(
        question,
        limit=3
    )

    context = build_context(results)

    prompt = f"""
You are an expert technical interviewer.

Candidate role:
{role}

Interview question:
{question}

Candidate answer:
{answer}

Relevant interview knowledge:
{context}

Evaluate the candidate's answer.

Return exactly this format:

SCORE: X/10

STRENGTHS:
- strength 1
- strength 2

WEAKNESSES:
- weakness 1
- weakness 2

FEEDBACK:
Short useful feedback for the candidate.

NEXT_DIFFICULTY:
easy / medium / hard

Be fair and evaluate the actual answer.
Do not invent things the candidate did not say.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


# ============================================================
# 5. FINAL INTERVIEW EVALUATION
# ============================================================

def final_evaluation(
    role,
    interview_history
):

    history_text = ""

    for i, item in enumerate(
        interview_history,
        1
    ):

        history_text += f"""
QUESTION {i}:
{item["question"]}

CANDIDATE ANSWER:
{item["answer"]}

INDIVIDUAL EVALUATION:
{item["evaluation"]}

--------------------------------------------------
"""

    prompt = f"""
You are a senior professional interviewer.

The candidate applied for:

{role}

The candidate completed a 5-question interview.

Here is the complete interview:

{history_text}

Create a final interview evaluation.

Return the evaluation in exactly this structure:

FINAL INTERVIEW EVALUATION
==========================

OVERALL SCORE:
X/10

TECHNICAL KNOWLEDGE:
X/10

PROBLEM SOLVING:
X/10

COMMUNICATION:
X/10

STRENGTHS:
- ...
- ...
- ...

WEAKNESSES:
- ...
- ...
- ...

OVERALL FEEDBACK:
Write a concise but useful overall assessment.

RECOMMENDATION:
Strong Hire / Hire / Consider / Weak Consider / Not Recommended

AREAS TO IMPROVE:
- ...
- ...
- ...

Do not invent information.
Base the evaluation only on the candidate's answers.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


# ============================================================
# STREAMLIT SESSION STATE
# ============================================================

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "role" not in st.session_state:
    st.session_state.role = ""

if "question_number" not in st.session_state:
    st.session_state.question_number = 0

if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []

if "interview_history" not in st.session_state:
    st.session_state.interview_history = []

if "previous_answer" not in st.session_state:
    st.session_state.previous_answer = None

if "previous_evaluation" not in st.session_state:
    st.session_state.previous_evaluation = None

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "current_evaluation" not in st.session_state:
    st.session_state.current_evaluation = None

if "final_result" not in st.session_state:
    st.session_state.final_result = None

if "answer_submitted" not in st.session_state:
    st.session_state.answer_submitted = False


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Interviewer")

st.write(
    "AI-powered interview system using "
    "Ollama, PostgreSQL, pgvector and Nomic embeddings."
)


# ============================================================
# START SCREEN
# ============================================================

if not st.session_state.interview_started:

    st.subheader("Start Your Interview")

    role = st.text_input(
        "What position are you applying for?"
    )

    if st.button(
        "🚀 Start Interview",
        use_container_width=True
    ):

        if not role.strip():

            st.warning(
                "Please enter the position you are applying for."
            )

        else:

            st.session_state.role = role

            st.session_state.interview_started = True

            st.session_state.question_number = 1

            st.session_state.asked_questions = []

            st.session_state.interview_history = []

            st.session_state.previous_answer = None

            st.session_state.previous_evaluation = None

            st.session_state.current_evaluation = None

            st.session_state.final_result = None

            st.session_state.answer_submitted = False

            # Generate first question
            with st.spinner(
                "AI is preparing your first question..."
            ):

                question = generate_question(
                    role=role,
                    previous_answer=None,
                    evaluation=None,
                    asked_questions=[]
                )

            st.session_state.current_question = question

            st.rerun()


# ============================================================
# INTERVIEW SCREEN
# ============================================================

else:

    st.subheader(
        f"Position: {st.session_state.role}"
    )

    st.progress(
        st.session_state.question_number / 5
    )

    st.write(
        f"### Question "
        f"{st.session_state.question_number} of 5"
    )

    # --------------------------------------------------------
    # Display current question
    # --------------------------------------------------------

    st.info(
        st.session_state.current_question
    )


    # --------------------------------------------------------
    # Candidate answer
    # --------------------------------------------------------

    if not st.session_state.answer_submitted:

        answer = st.text_area(
            "Your Answer",
            height=200,
            placeholder="Type your answer here..."
        )

        if st.button(
            "📤 Submit Answer",
            use_container_width=True
        ):

            if not answer.strip():

                st.warning(
                    "Please provide an answer."
                )

            else:

                with st.spinner(
                    "AI is evaluating your answer..."
                ):

                    evaluation = evaluate_answer(
                        st.session_state.role,
                        st.session_state.current_question,
                        answer
                    )

                # Save answer and evaluation
                st.session_state.interview_history.append(
                    {
                        "question":
                            st.session_state.current_question,

                        "answer":
                            answer,

                        "evaluation":
                            evaluation
                    }
                )

                st.session_state.previous_answer = answer

                st.session_state.previous_evaluation = evaluation

                st.session_state.current_evaluation = evaluation

                st.session_state.answer_submitted = True

                st.rerun()


    # ========================================================
    # SHOW INDIVIDUAL EVALUATION
    # ========================================================

    else:

        st.subheader(
            "🧠 Individual Answer Evaluation"
        )

        st.markdown(
            st.session_state.current_evaluation
        )


        # ====================================================
        # IF QUESTIONS REMAIN
        # ====================================================

        if st.session_state.question_number < 5:

            if st.button(
                "➡️ Next Question",
                use_container_width=True
            ):

                next_question_number = (
                    st.session_state.question_number + 1
                )

                with st.spinner(
                    "AI is preparing the next question..."
                ):

                    question = generate_question(
                        role=st.session_state.role,

                        previous_answer=
                            st.session_state.previous_answer,

                        evaluation=
                            st.session_state.previous_evaluation,

                        asked_questions=
                            st.session_state.asked_questions
                    )

                st.session_state.asked_questions.append(
                    question
                )

                st.session_state.current_question = question

                st.session_state.question_number = (
                    next_question_number
                )

                st.session_state.answer_submitted = False

                st.session_state.current_evaluation = None

                st.rerun()


        # ====================================================
        # AFTER QUESTION 5
        # ====================================================

        else:

            st.success(
                "🎉 You have completed all 5 questions!"
            )

            if st.session_state.final_result is None:

                if st.button(
                    "📊 Generate Final Evaluation",
                    use_container_width=True
                ):

                    with st.spinner(
                        "AI is generating your final evaluation..."
                    ):

                        final_result = final_evaluation(
                            st.session_state.role,
                            st.session_state.interview_history
                        )

                    st.session_state.final_result = final_result

                    st.rerun()


# ============================================================
# FINAL RESULT
# ============================================================

if st.session_state.final_result is not None:

    st.divider()

    st.header(
        "🏆 Final Interview Evaluation"
    )

    st.markdown(
        st.session_state.final_result
    )

    st.divider()

    if st.button(
        "🔄 Start New Interview",
        use_container_width=True
    ):

        st.session_state.interview_started = False

        st.session_state.role = ""

        st.session_state.question_number = 0

        st.session_state.asked_questions = []

        st.session_state.interview_history = []

        st.session_state.previous_answer = None

        st.session_state.previous_evaluation = None

        st.session_state.current_question = None

        st.session_state.current_evaluation = None

        st.session_state.final_result = None

        st.session_state.answer_submitted = False

        st.rerun()