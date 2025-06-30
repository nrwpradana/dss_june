import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load data and models
data = pd.read_csv("US E-commerce records 2020.csv")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
llm = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Preprocess and index data
processed_data = ["Order Date: {} ... Profit: {}".format(row['Order Date'], row['Profit']) for _, row in data.iterrows()]
embeddings = embedder.encode(processed_data)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Streamlit UI
st.title("E-commerce Data Insights with RAG")
query = st.text_input("Enter your query:")
if query:
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k=5)
    retrieved_entries = [processed_data[i] for i in I[0]]
    prompt = f"Query: {query}\nContext: {retrieved_entries}\nProvide a concise response."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("Response:", response)
    st.write("Retrieved Data:", retrieved_entries)
