import ollama
import psycopg2

try:
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5433,
        database="ragdb",
        user="postgres",
        password="postgres"
    )
    print("Connected successfully!")

except psycopg2.Error as e:
    print("POSTGRESQL ERROR:")
    print(str(e))
cur = conn.cursor()

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

print("Table created successfully!")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)
# ============================================================
# 1. SEARCH QUESTIONS USING NOMIC + PGVECTOR
# ============================================================

def search_questions(query, limit=5):

    print("\n🔍 Searching PostgreSQL + pgvector...")

    query_embedding = model.encode(query).tolist()

    print(f"   ✓ Nomic embedding created ({len(query_embedding)} dimensions)")

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

    print(f"   ✓ Retrieved {len(results)} relevant questions")

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

def generate_question(role, previous_answer=None, evaluation=None, asked_questions=None):

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

    results = search_questions(search_query, limit=5)

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

def evaluate_answer(role, question, answer):

    print("\n🧠 Evaluating your answer...")

    results = search_questions(question, limit=3)

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

def final_evaluation(role, interview_history):

    print("\n")
    print("=" * 65)
    print("GENERATING FINAL INTERVIEW EVALUATION")
    print("=" * 65)

    history_text = ""

    for i, item in enumerate(interview_history, 1):

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
# 6. START 5-QUESTION INTERVIEW
# ============================================================

def start_interview():

    print("\n")
    print("=" * 65)
    print("                    AI INTERVIEWER")
    print("=" * 65)

    print("\nAI: Welcome! I will conduct a 5-question interview.")

    # --------------------------------------------------------
    # Ask role
    # --------------------------------------------------------

    role = input(
        "\nAI: What position are you applying for?\nYou: "
    )

    if role.lower() in ["exit", "quit"]:
        print("\nInterview ended.")
        return

    print(f"\nAI: Great. This interview will focus on: {role}")

    # --------------------------------------------------------
    # Interview variables
    # --------------------------------------------------------

    MAX_QUESTIONS = 5

    asked_questions = []

    interview_history = []

    previous_answer = None

    previous_evaluation = None

    # --------------------------------------------------------
    # Five-question interview
    # --------------------------------------------------------

    for question_number in range(1, MAX_QUESTIONS + 1):

        print("\n")
        print("=" * 65)
        print(f"QUESTION {question_number} OF {MAX_QUESTIONS}")
        print("=" * 65)

        # Generate question
        question = generate_question(
            role=role,
            previous_answer=previous_answer,
            evaluation=previous_evaluation,
            asked_questions=asked_questions
        )

        asked_questions.append(question)

        print("\nAI INTERVIEWER:")
        print(question)

        # ----------------------------------------------------
        # Get candidate answer
        # ----------------------------------------------------

        answer = input("\nYou: ")

        if answer.lower() in ["exit", "quit"]:

            print("\nAI: Interview ended early.")
            return

        while not answer.strip():

            print("AI: Please provide an answer.")

            answer = input("\nYou: ")

        # ----------------------------------------------------
        # Evaluate answer
        # ----------------------------------------------------

        evaluation = evaluate_answer(
            role,
            question,
            answer
        )

        print("\n")
        print("-" * 65)
        print("INDIVIDUAL ANSWER EVALUATION")
        print("-" * 65)

        print(evaluation)

        # ----------------------------------------------------
        # Save interview data
        # ----------------------------------------------------

        interview_history.append({
            "question": question,
            "answer": answer,
            "evaluation": evaluation
        })

        # Prepare for next question
        previous_answer = answer
        previous_evaluation = evaluation

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    final_result = final_evaluation(
        role,
        interview_history
    )

    print("\n")
    print("=" * 65)
    print("                    FINAL RESULT")
    print("=" * 65)

    print(final_result)

    print("\n")
    print("=" * 65)
    print("                INTERVIEW COMPLETE")
    print("=" * 65)


# ============================================================
# RUN
# ============================================================

start_interview()