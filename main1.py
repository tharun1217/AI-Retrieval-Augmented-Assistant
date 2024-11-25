import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv








def load_embedding_and_chunk(embeddings_file,chunk_file):
    embeddings = np.load(embeddings_file)
    with open(chunk_file, "r", encoding="utf-8") as file:
        chunks = file.read().split("\n\n")
    return embeddings, chunks

def load_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_faiss_index(query,model,index,chunks,top_k=5):
    query_embedding = model.encode([query])
    distance,indices = index.search(query_embedding,k=top_k)
    return "\n\n".join([chunks[i] for i in indices[0]])

def generate_response(query, retrieved_text):
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model="llama-3.1-70b-versatile",
    )

    template = [
        ("system", "You are a highly skilled text extraction and information retrieval system. Your task is to find the related answer according to the question."),
        ("user", "{query}\n\nHere are the answers:\n{main_answer}"),
    ]
    prompt = ChatPromptTemplate(template)
    chain = prompt | llm | StrOutputParser()

   
    response = chain.invoke({'query': query, 'main_answer': retrieved_text})
    return response

if __name__ == "__main__":
    embeddings_file = "embeddings.npy"
    chunks_file = "text_chunk.txt"
    embeddings, chunks = load_embedding_and_chunk(embeddings_file, chunks_file)

    faiss_index = load_faiss_index(embeddings)
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query = "hello"

    retrieved_text = search_faiss_index(query, model, faiss_index, chunks, top_k=5)
    # print("Retrieved text:")
    # print(retrieved_text)

   
    response = generate_response(query, retrieved_text)
    print("\nLLM Response:")
    print(response)
    