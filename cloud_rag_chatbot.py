# cloud_rag_chatbot.py (使用 HuggingFace 替代 OpenAI)

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint  # 使用 HuggingFace
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from document_loader import process_documents
from embedding_store import create_vector_store, load_vector_store

# 不再需要從.env文件加載環境變量
# load_dotenv()

def setup_rag_system(rebuild_vector_store=False, model_name="google/flan-t5-large", temperature=0.2, k=2):
    """
    設置RAG系統，包括文檔處理和向量存儲。
    
    Args:
        rebuild_vector_store: 是否從頭重建向量存儲
        model_name: 要使用的模型名稱
        temperature: 溫度參數
        k: 檢索的文檔數量
    
    Returns:
        RetrievalQA鏈
    """
    # 檢查向量存儲是否存在
    if rebuild_vector_store or not os.path.exists("./chroma_db"):
        print("構建向量存儲...")
        # 處理文檔
        chunks = process_documents()
        # 創建向量存儲
        vector_store = create_vector_store(chunks)
    else:
        print("加載現有向量存儲...")
        # 加載現有向量存儲
        vector_store = load_vector_store()
    
    # 創建檢索器
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # 創建自定義提示模板
    template = """
    你是一個專業的台灣法律助手。使用以下上下文來回答最後的問題。
    如果你不知道答案，就直說不知道，不要試圖編造答案。
    請用繁體中文（台灣用語）回答問題，避免使用簡體中文詞彙。
    
    上下文:
    {context}
    
    問題: {question}
    
    答案:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # 初始化 HuggingFace 語言模型
    # 使用支持中文的模型，如 BLOOM 或 mT5
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        temperature=temperature,
        huggingfacehub_api_token=st.secrets["HUGGINGFACE_API_TOKEN"]  # 使用 Streamlit secrets
    )
    
    # 創建RAG鏈
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return rag_chain

# 其餘函數保持不變
def ask_question(rag_chain, question):
    """向RAG系統提問"""
    response = rag_chain.invoke({"query": question})
    answer = response["result"]
    source_docs = response["source_documents"]
    return answer, source_docs