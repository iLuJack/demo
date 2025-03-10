# embedding_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

def create_vector_store(chunks):
    """
    創建向量存儲以便高效檢索。
    """
    print("創建向量存儲...")
    
    # Fix the deprecation warning by using the correct import
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # 使用繁體中文優化的嵌入模型
    embedding_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese"
    )
    
    # Use FAISS instead of Chroma
    from langchain_community.vectorstores import FAISS
    
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    
    # Save the FAISS index
    vector_store.save_local("faiss_index")
    
    print(f"向量存儲已創建，包含 {len(chunks)} 個文檔")
    return vector_store

def load_vector_store():
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese"
    )
    
    return FAISS.load_local("faiss_index", embedding_model)

if __name__ == "__main__":
    # 測試向量存儲創建
    from document_loader import process_documents
    
    # 處理文檔
    chunks = process_documents()
    
    # 創建向量存儲
    vector_store = create_vector_store(chunks)
    
    # 測試簡單查詢
    results = vector_store.similarity_search("竊盜罪的處罰是什麼？", k=2)
    
    print("\n測試查詢結果:")
    for doc in results:
        print(f"\n來源: {doc.metadata['source']}")
        print(f"內容: {doc.page_content[:150]}...")