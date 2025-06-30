import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
import io

# Streamlit UI
st.title("E-commerce Data Insights with RAG")
st.write("Upload your CSV file to analyze e-commerce data.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Read and process data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
    processed_data = [
        f"Order Date: {row['Order Date']}, Row ID: {row['Row ID']}, Order ID: {row['Order ID']}, "
        f"Ship Mode: {row['Ship Mode']}, Customer ID: {row['Customer ID']}, ..., Profit: {row['Profit']}"
        for _, row in data.iterrows()
    ]
    st.write("Processed Data Sample:", processed_data[:2])

    # Initialize models and index
    if "index" not in st.session_state:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode(processed_data)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        st.session_state.index = index
        st.session_state.processed_data = processed_data
        st.session_state.embedder = embedder

    # Load LLM
    if "llm" not in st.session_state:
        st.session_state.llm = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
        st.session_state.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

    # Query input
    query = st.text_input("Enter your query:")
    if query:
        query_embedding = st.session_state.embedder.encode([query])
        D, I = st.session_state.index.search(query_embedding, k=5)
        retrieved_entries = [st.session_state.processed_data[i] for i in I[0]]
        prompt = f"Query: {query}\nContext: {retrieved_entries}\nProvide a concise response."
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt")
        outputs = st.session_state.llm.generate(**inputs, max_length=200)
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("**Response**:", response)
        st.write("**Retrieved Data**:", retrieved_entries)

        # Optional visualization
        category_sales = data.groupby("Category")["Sales"].sum().reset_index()
        st.bar_chart(category_sales.set_index("Category"))
