# main.py
import streamlit as st
from vectorstore import get_vectorstore
from groq_llm import generate_response

st.set_page_config(page_title="🧠 RAG App", layout="wide")
st.title("🧠 Retrieval-Augmented Generation (RAG) App")

query = st.text_input("Ask something based on your documents...")

if query:
    vectordb = get_vectorstore()
    docs = vectordb.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = f"""You are an assistant. Use the context below to answer the question. If the query is out of context just say Sorry, I don't Know

    CONTEXT:
    {context}

    QUESTION:
    {query}
    """

    with st.spinner("Thinking..."):
        response = generate_response(final_prompt)

    st.subheader("Answer:")
    st.write(response)
