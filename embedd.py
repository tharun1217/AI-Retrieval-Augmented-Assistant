from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter



def extract_text(pdf_path):
    text =''
    with open(pdf_path,'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            text+=page_text
    return text     

def split_text(text,chunksize=1000,chunkoverlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize,chunk_overlap = chunkoverlap)
    return text_splitter.split_text(text)

def generate_embeddings(chunks):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(chunks)

if __name__=="__main__":
    pdf_file = r"D:\rag\A_Brief_Introduction_To_AI.pdf"
    text = extract_text(pdf_file)
    chunks = split_text(text)
    
    with open("text_chunk.txt","w",encoding='utf-8') as file:
        file.write("\n\n".join(chunks))
    import numpy as np   
    embeddings = generate_embeddings(chunks)
    np.save("embeddings.npy", embeddings)
    print("Embeddings saved to 'embeddings.npy'.")                
   