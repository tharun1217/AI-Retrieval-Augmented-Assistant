import faiss
import numpy as np

DEFAULT_FAISS_INDEX_FILE = "faiss_index"

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_faiss(index,file_name):
    faiss.write_index(index,file_name)
    
if __name__ == "__main__":
    embeddings = np.load("embeddings.npy")
    index = create_faiss_index(embeddings)
    
    save_faiss(index,"faiss_index")
    print("FAISS Index Saved")