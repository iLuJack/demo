# embedding_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

def create_vector_store(chunks, persist_directory="/tmp/chroma_db"):
    """
    從文檔塊創建向量存儲。
    
    Args:
        chunks: 文檔塊列表
        persist_directory: 存儲向量數據庫的目錄
    
    Returns:
        Chroma向量存儲實例
    """
    # 檢查是否存在舊的向量存儲
    if os.path.exists(persist_directory):
        print(f"檢測到現有向量存儲，正在刪除 {persist_directory} 目錄...")
        # 刪除舊的向量存儲以避免維度不匹配問題
        shutil.rmtree(persist_directory)
        print("舊向量存儲已刪除")
    
    # 初始化嵌入模型
    # 使用支援繁體中文的公開模型
    embedding_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese-paraphrase"  # 支持繁體中文的模型
    )
    
    print("創建向量存儲...")
    
    # 創建持久化Chroma向量存儲
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    # 將向量存儲持久化到磁盤
    vector_store.persist()
    print(f"向量存儲已創建並保存到 {persist_directory}")
    
    return vector_store

def load_vector_store(persist_directory="./chroma_db"):
    """
    從磁盤加載現有的向量存儲。
    
    Args:
        persist_directory: 保存向量存儲的目錄
    
    Returns:
        Chroma向量存儲實例
    """
    # 檢查目錄是否存在
    if not os.path.exists(persist_directory):
        raise ValueError(f"向量存儲目錄 {persist_directory} 不存在")
    
    # 初始化嵌入模型（必須與創建時使用的相同）
    embedding_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese-paraphrase"  # 支持繁體中文的模型
    )
    
    # 加載向量存儲
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    
    print(f"已從 {persist_directory} 加載向量存儲")
    return vector_store

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