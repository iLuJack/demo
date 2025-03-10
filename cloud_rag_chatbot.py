# cloud_rag_chatbot.py (使用 HuggingFace 替代 OpenAI)

import os
import time
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFacePipeline  # 添加本地模型支持
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.fake import FakeListLLM  # 用於本地備用模型

from document_loader import process_documents
from embedding_store import create_vector_store, load_vector_store

# 不再需要從.env文件加載環境變量
# load_dotenv()

def setup_rag_system(rebuild_vector_store=False, model_name=None, temperature=0.1, k=2, api_token=None, max_retries=3, use_local_fallback=True):
    """
    設置RAG系統，包括文檔處理和向量存儲。
    
    Args:
        rebuild_vector_store: 是否從頭重建向量存儲
        model_name: 要使用的模型名稱 (如果為None，將使用預設模型列表)
        temperature: 溫度參數
        k: 檢索的文檔數量
        api_token: HuggingFace API 令牌，如果為 None，則嘗試從環境變量獲取
        max_retries: 嘗試連接 HuggingFace API 的最大次數
        use_local_fallback: 是否在API不可用時使用本地模型
    
    Returns:
        RetrievalQA鏈或(模式標記, 檢索器)元組
    """
    # 更新模型列表，使用更可靠的模型
    models_to_try = [
        "google/flan-t5-large",      # 多語言模型，支持中文
        "google/flan-t5-base",       # 較小的模型，更可能可用
        "facebook/bart-large-cnn",   # 另一個可能的備選
        "google/flan-t5-small",      # 最小的模型，最可能可用
        "facebook/bart-base"         # 備用選項
    ]
    
    # 如果指定了模型，將其放在列表最前面
    if model_name and model_name not in models_to_try:
        models_to_try.insert(0, model_name)
    
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
    
    # 創建自定義提示模板 - 更明確的指示
    template = """
    你是一個專業的台灣法律助手。使用以下上下文來回答最後的問題。
    如果你不知道答案，就直說不知道，不要試圖編造答案。
    請用繁體中文（台灣用語）回答問題，避免使用簡體中文詞彙。
    
    上下文:
    {context}
    
    問題: {question}
    
    請提供詳細的回答:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # 獲取 API 令牌，優先使用傳入的參數
    if api_token is None:
        # 嘗試從環境變量獲取
        api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
    
    # 嘗試初始化 HuggingFace 語言模型
    llm = None
    
    # 首先嘗試API模型
    for attempt, model in enumerate(models_to_try):
        for retry in range(max_retries):
            try:
                print(f"嘗試使用模型: {model} (嘗試 {retry+1}/{max_retries})")
                
                llm = HuggingFaceEndpoint(
                    repo_id=model,
                    huggingfacehub_api_token=api_token,
                    model_kwargs={
                        "temperature": temperature,
                        "max_new_tokens": 250,  # 增加生成的令牌數
                        "top_p": 0.9,
                    }
                )
                
                # 測試模型
                test_response = llm.invoke("請簡單介紹台灣的法律制度")
                print(f"成功連接到模型: {model}")
                print(f"測試回應: {test_response[:50]}...")  # 打印部分回應以確認
                
                if hasattr(st, 'session_state'):
                    st.session_state.current_model = f"{model} (API)"
                break
                
            except Exception as e:
                print(f"連接到模型 {model} 時出錯: {e}")
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2
                    print(f"等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                else:
                    print(f"無法連接到模型 {model}，嘗試下一個模型...")
        
        if llm is not None:
            break
    
    # 如果API模型都失敗，嘗試使用本地模型
    if llm is None and use_local_fallback:
        try:
            print("嘗試使用本地模型作為備用...")
            
            # 預定義一些常見法律問題的回答
            responses = {
                "台灣的法律制度": "台灣的法律制度主要基於大陸法系，同時也受到英美法系的影響。司法體系包括最高法院、高等法院和地方法院三級制度。",
                "民法": "台灣民法規範私人間的權利義務關係，包括人格權、物權、債權、親屬和繼承等部分。",
                "刑法": "台灣刑法規定犯罪行為及其處罰，保護社會安全和個人權益。",
                "憲法": "中華民國憲法是台灣的根本大法，規定國家組織、人民權利義務等基本原則。",
                "法院": "台灣的法院系統包括普通法院、行政法院、智慧財產法院等，負責審理各類案件。",
                "律師": "律師是具有法律專業知識，經考試及格並依法取得律師資格的法律專業人員。",
                "訴訟": "訴訟是解決爭議的法律程序，包括民事訴訟、刑事訴訟和行政訴訟等。",
            }
            
            # 創建一個簡單的本地模型
            llm = FakeListLLM(responses=[
                "根據提供的資料，" + responses.get(keyword, "我無法提供具體答案，請查看來源文檔以獲取相關信息。")
                for keyword in responses.keys()
            ])
            
            print("成功初始化本地備用模型")
            
            if hasattr(st, 'session_state'):
                st.session_state.current_model = "本地備用模型"
                
        except Exception as e:
            print(f"初始化本地模型時出錯: {e}")
    
    # 如果無法初始化任何模型，返回一個特殊標記和檢索器
    if llm is None:
        print("警告: 無法初始化任何模型，將使用純文檔搜索模式")
        if hasattr(st, 'session_state'):
            st.session_state.current_model = "純文檔搜索模式"
        return ("DOCUMENTS_ONLY", retriever)
    
    # 創建RAG鏈
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return rag_chain

# 添加純文檔搜索函數
def search_documents_only(retriever, question):
    """
    當模型不可用時，僅使用文檔檢索功能
    """
    docs = retriever.get_relevant_documents(question)
    
    # 構建一個簡單的回答
    if docs:
        answer = f"【純文檔搜索模式】我找到了以下與您問題相關的資訊：\n\n"
        for i, doc in enumerate(docs):
            answer += f"資料來源 {i+1}: {doc.metadata.get('source', '未知來源')}\n"
            answer += f"{doc.page_content[:300]}...\n\n"
    else:
        answer = "【純文檔搜索模式】抱歉，我無法找到與您問題相關的資訊。請嘗試使用不同的關鍵詞。"
    
    return answer, docs

# 修改問答函數以支持不同模式
def ask_question(rag_chain_or_retriever, question):
    """向RAG系統提問或僅搜索文檔"""
    try:
        # 檢查是否處於純文檔搜索模式
        if isinstance(rag_chain_or_retriever, tuple) and rag_chain_or_retriever[0] == "DOCUMENTS_ONLY":
            _, retriever = rag_chain_or_retriever
            return search_documents_only(retriever, question)
        
        # 正常 RAG 模式
        response = rag_chain_or_retriever.invoke({"query": question})
        answer = response.get("result", "")
        
        # 檢查回答是否為空
        if not answer or answer.strip() == "":
            answer = "抱歉，我無法生成回答。請嘗試重新表述您的問題，或者查看來源文檔以獲取相關信息。"
        
        source_docs = response.get("source_documents", [])
        return answer, source_docs
    except Exception as e:
        print(f"生成回答時出錯: {e}")
        # 如果在生成回答時出錯，嘗試使用純文檔搜索
        if hasattr(rag_chain_or_retriever, "retriever"):
            retriever = rag_chain_or_retriever.retriever
            print("退回到純文檔搜索模式")
            return search_documents_only(retriever, question)
        return f"生成回答時出錯: {str(e)}", []

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