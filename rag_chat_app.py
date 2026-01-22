from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# ----- load and prepare knowledge base -----
with open("cricket_knowledge.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

chunks = [para.strip() for para in full_text.split("\n\n") if para.strip()]
doc_embeddings = embeddings.embed_documents(chunks)

def rag_answer(question: str):
    query_embedding = embeddings.embed_query(question)
    scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    best_index = int(scores.argmax())
    best_doc = chunks[best_index]

    prompt = f"""You are a helpful assistant.
Use ONLY the information in the CONTEXT to answer the QUESTION.

CONTEXT:
{best_doc}

QUESTION:
{question}
"""
    response = llm.invoke(prompt)
    return best_doc, response.content

# ----- Streamlit UI with history -----
st.title("Cricket RAG Chatbot")

# initialize chat history in session_state
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer)

user_q = st.text_input("Ask a question about the players:")

if st.button("Ask") and user_q.strip():
    context, answer = rag_answer(user_q)
    st.session_state.history.append((user_q, answer))

    st.subheader("Context used for this answer")
    st.write(context)

st.subheader("Chat history")
for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")
