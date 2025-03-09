# document_loader.py

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents(directory_path):
    """
    從指定目錄加載所有文本文檔。
    """
    documents = []
    
    # 獲取目錄中的所有txt文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                # 加載文檔，確保使用utf-8編碼以支持繁體中文
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
                print(f"已加載文檔: {filename}")
            except Exception as e:
                print(f"加載 {filename} 時出錯: {e}")
    
    return documents

def split_documents(documents):
    """
    將文檔分割成更小的塊以便更好地處理。
    使用適合繁體中文的分割方式。
    """
    # 創建文本分割器（為繁體中文優化）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # 繁體中文字符通常需要較小的塊大小
        chunk_overlap=80,
        length_function=len,
        # 繁體中文標點符號
        separators=["\n\n", "\n", "。", "！", "？", "，", "；", "：", "　", " ", ""]
    )
    
    # 將文檔分割成塊
    chunks = text_splitter.split_documents(documents)
    print(f"將 {len(documents)} 個文檔分割成 {len(chunks)} 個塊")
    
    return chunks

def process_documents(directory_path="./data"):
    """
    加載並處理數據目錄中的所有文檔。
    """
    # 加載文檔
    documents = load_documents(directory_path)
    
    # 分割成塊
    chunks = split_documents(documents)
    
    return chunks

if __name__ == "__main__":
    # 測試文檔加載和處理
    chunks = process_documents()
    
    # 打印樣本塊以驗證
    if chunks:
        print("\n樣本塊內容:")
        print(f"內容: {chunks[0].page_content[:150]}...")
        print(f"來源: {chunks[0].metadata['source']}")
    else:
        print("沒有處理任何文檔。")