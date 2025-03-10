# cloud_rag_chatbot.py (使用 HuggingFace 替代 OpenAI)

import os
import time
import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from document_loader import process_documents
from embedding_store import create_vector_store, load_vector_store

# 不再需要從.env文件加載環境變量
# load_dotenv()

def setup_rag_system(rebuild_vector_store=False, model_name="bigscience/bloom-1b7", temperature=0.1, k=2, api_token=None, max_retries=3):
    """
    設置RAG系統，包括文檔處理和向量存儲。
    
    Args:
        rebuild_vector_store: 是否從頭重建向量存儲
        model_name: 要使用的模型名稱 (默認為 bloom-1b7)
        temperature: 溫度參數
        k: 檢索的文檔數量
        api_token: HuggingFace API 令牌，如果為 None，則嘗試從環境變量獲取
        max_retries: 嘗試連接 HuggingFace API 的最大次數
    
    Returns:
        RetrievalQA鏈
    """
    # 強制使用 bloom-1b7 模型
    primary_model = "bigscience/bloom-1b7"
    # 備用模型列表，如果主要模型不可用
    backup_models = [
        "google/flan-t5-large",  # 多語言模型
        "facebook/bart-large-cnn",  # 另一個可能的備選
        "google/flan-t5-base"  # 較小的模型，更可能可用
    ]
    
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
        try:
            vector_store = load_vector_store()
            print("向量存儲加載成功")
        except Exception as e:
            print(f"加載向量存儲時出錯: {e}")
            print("重新構建向量存儲...")
            chunks = process_documents()
            vector_store = create_vector_store(chunks)
    
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
    
    # 獲取 API 令牌，優先使用傳入的參數
    if api_token is None:
        # 嘗試從環境變量獲取
        api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
    
    # 嘗試初始化 HuggingFace 語言模型，如果失敗則嘗試備用模型
    llm = None
    models_to_try = [primary_model] + backup_models
    
    for attempt, model in enumerate(models_to_try):
        for retry in range(max_retries):
            try:
                print(f"嘗試使用模型: {model} (嘗試 {retry+1}/{max_retries})")
                llm = HuggingFaceHub(
                    repo_id=model,
                    huggingfacehub_api_token=api_token,
                    model_kwargs={
                        "temperature": temperature,
                        "max_new_tokens": 250,  # 確保不超過 API 限制
                        "top_p": 0.9,
                    }
                )
                
                # 測試模型是否可用
                _ = llm.invoke("測試連接")
                print(f"成功連接到模型: {model}")
                break
                
            except Exception as e:
                print(f"連接到模型 {model} 時出錯: {e}")
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2  # 指數退避
                    print(f"等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                else:
                    print(f"無法連接到模型 {model}，嘗試下一個模型...")
        
        if llm is not None:
            break
    
    if llm is None:
        raise Exception("無法連接到任何 HuggingFace 模型。請稍後再試。")
    
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

# Add this code at the bottom of the file
if __name__ == "__main__":
    print("初始化 RAG 系統...")
    rag_chain = setup_rag_system()
    print("RAG 系統已準備就緒！")
    
    # Interactive mode for testing
    while True:
        user_question = input("\n請輸入您的法律問題 (輸入 'exit' 退出): ")
        if user_question.lower() == 'exit':
            break
            
        print("\n正在處理您的問題...")
        answer, sources = ask_question(rag_chain, user_question)
        
        print("\n回答:")
        print(answer)
        
        print("\n來源文檔:")
        for i, doc in enumerate(sources):
            print(f"文檔 {i+1}:")
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
            print()