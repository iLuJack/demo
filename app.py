# app.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import shutil
import time
import threading
import queue

# Adjust import paths for Streamlit Cloud
from document_loader import process_documents
from embedding_store import create_vector_store, load_vector_store
from cloud_rag_chatbot import setup_rag_system, ask_question

# 設置頁面配置和標題
st.set_page_config(
    page_title="法律小幫手",
    page_icon="⚖️",
    layout="wide"
)

st.title("臺大管院測試法律小幫手 ⚖️")
st.markdown("*基於 RAG 技術的台灣法律問答系統*")

# 初始化 session state 變量
if "current_model" not in st.session_state:
    st.session_state.current_model = "正在連接..."

# 側邊欄配置
with st.sidebar:
    st.header("系統設置")
    
    # 重建向量存儲選項
    rebuild_db = st.checkbox("重建向量數據庫", value=False, 
                           help="選中此項將重新處理文檔並創建新的向量存儲")
    
    # 顯示來源文檔選項
    show_sources = st.checkbox("顯示來源文檔", value=True,
                             help="顯示回答的來源文檔")
    
    # 顯示當前使用的模型
    st.info(f"使用模型: {st.session_state.current_model}")
    
    # 溫度設置
    temperature = st.slider("溫度", min_value=0.0, max_value=1.0, value=0.2, step=0.1,
                          help="較低的值使回答更確定，較高的值使回答更多樣化")
    
    # 檢索文檔數量
    k_docs = st.slider("檢索文檔數量", min_value=1, max_value=5, value=2, step=1,
                      help="從向量存儲中檢索的文檔數量")
    
    # 添加重置按鈕
    if st.button("重置聊天"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("### 關於")
    st.markdown("此應用使用 HuggingFace API、LangChain 和 Streamlit 構建")
    st.markdown("© 2025 臺大管院測試法律助手")

# 檢查 API 密鑰
if not st.secrets.get("HUGGINGFACE_API_TOKEN"):
    st.warning("未設置 HUGGINGFACE_API_TOKEN 環境變量，將嘗試使用本地模型或純文檔搜索模式")

# 初始化聊天歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

# 檢查預構建的向量存儲
if not os.path.exists("./chroma_db") and os.path.exists("./pre_built_chroma_db"):
    with st.spinner("正在加載預構建的向量數據庫..."):
        # 複製預構建的數據庫到預期位置
        shutil.copytree("./pre_built_chroma_db", "./chroma_db")

# 初始化 RAG 系統
if "rag_chain" not in st.session_state or rebuild_db:
    with st.spinner("正在加載 RAG 系統..."):
        try:
            # 獲取 API 令牌
            api_token = st.secrets.get("HUGGINGFACE_API_TOKEN")
            
            # 設置 RAG 系統
            rag_chain = setup_rag_system(
                rebuild_vector_store=rebuild_db,
                temperature=temperature,
                k=k_docs,
                api_token=api_token,
                max_retries=2  # 減少重試次數以加快加載
            )
            
            # 保存到 session state
            st.session_state.rag_chain = rag_chain
            
        except Exception as e:
            st.error(f"無法加載 RAG 系統: {str(e)}")
            st.stop()

# 獲取 RAG 系統
rag_chain = st.session_state.rag_chain

# 創建聊天容器
chat_container = st.container()

# 顯示聊天歷史
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 顯示來源（如果有且用戶選擇顯示）
            if show_sources and message["role"] == "assistant" and i < len(st.session_state.sources):
                sources = st.session_state.sources[i]
                if sources:
                    with st.expander("查看來源文檔"):
                        for j, source in enumerate(sources):
                            st.markdown(f"**來源 {j+1}**: {source['source']}")
                            st.markdown(f"```\n{source['content'][:300]}...\n```")

# 獲取用戶輸入
user_input = st.chat_input("請輸入您的法律問題...")

# 處理用戶輸入
if user_input:
    # 添加用戶消息到聊天歷史
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 顯示用戶消息
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 顯示助手消息
    with st.chat_message("assistant"):
        # 創建一個空的佔位符
        message_placeholder = st.empty()
        
        # 顯示思考中的指示
        thinking_msg = st.empty()
        thinking_msg.markdown("*思考中...*")
        
        # 獲取回答
        try:
            # 使用線程和隊列來處理可能的超時
            result_queue = queue.Queue()
            
            def run_query():
                try:
                    result = ask_question(rag_chain, user_input)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))
            
            # 啟動線程
            query_thread = threading.Thread(target=run_query)
            query_thread.start()
            
            # 等待結果，最多30秒
            start_time = time.time()
            max_wait_time = 30  # 秒
            
            while time.time() - start_time < max_wait_time:
                if not result_queue.empty():
                    status, result = result_queue.get()
                    break
                time.sleep(0.1)
            else:
                # 超時
                status = "timeout"
                result = ("查詢超時。系統將嘗試使用純文檔搜索模式。", [])
            
            # 處理結果
            if status == "success":
                answer, source_docs = result
            else:
                # 如果出錯或超時，顯示錯誤消息
                answer = f"發生錯誤: {result}" if status == "error" else result[0]
                source_docs = []
            
            # 顯示回答
            message_placeholder.markdown(answer)
            thinking_msg.empty()
            
            # 處理來源文檔
            if source_docs and show_sources:
                sources_data = []
                with st.expander("查看來源文檔"):
                    for i, doc in enumerate(source_docs):
                        source_name = doc.metadata['source'] if hasattr(doc, 'metadata') and 'source' in doc.metadata else "未知來源"
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        
                        st.markdown(f"**來源 {i+1}**: {source_name}")
                        st.markdown(f"```\n{content[:300]}...\n```")
                        sources_data.append({
                            "source": source_name,
                            "content": content
                        })
                
                # 保存來源以便歷史顯示
                st.session_state.sources.append(sources_data)
            else:
                st.session_state.sources.append([])
            
            # 添加助手消息到聊天歷史
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = f"發生錯誤: {str(e)}"
            message_placeholder.error(error_msg)
            thinking_msg.empty()
            
            # 添加錯誤消息到聊天歷史
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.session_state.sources.append([])