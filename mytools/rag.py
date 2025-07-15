import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor

def load_env():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("⚠️ Vui lòng thiết lập biến môi trường OPENAI_API_KEY trong .env")
    return api_key

def get_embedding_and_vectorstore():
    # Khởi tạo embedding và vectorstore song song
    with ThreadPoolExecutor() as executor:
        future_embedding = executor.submit(OpenAIEmbeddings)
        embedding = future_embedding.result()
        future_vectorstore = executor.submit(
            lambda: Chroma(
                persist_directory="chromadb",
                embedding_function=embedding,
            )
        )
        vectorstore = future_vectorstore.result()
    return embedding, vectorstore

QA_PROMPT = PromptTemplate.from_template("""
    Bạn là một hướng dẫn viên du lịch thông minh. Trả lời câu hỏi của người dùng bằng tiếng Việt, dựa trên thông tin từ dữ liệu địa điểm bên dưới.

    Nếu người dùng cần gợi ý, hãy đưa ra 5 địa điểm phù hợp nhất. Mỗi gợi ý trình bày theo mẫu:
                                       
    Tên: (Tên địa điểm)  
    Loại: (Loại địa điểm: quán ăn, cafe, khách sạn...)  
    Khu vực: (Phường, Quận)  
    Mô tả: (Mô tả ngắn gọn, hấp dẫn, nổi bật)  
    Thời gian mở đóng: (Open-Close)/(Checkin/Checkout)
    Thời gian gợi ý ở lại: (Duration_suggested_min) <Hotel thì bỏ qua>
    ---

    Nếu người dùng yêu cầu chi tiết hơn, bạn có thể mô tả sâu hơn về các dịch vụ, giờ mở cửa, phù hợp với nhóm nào, v.v.

    Câu hỏi của người dùng: {question}  
    Dữ liệu địa điểm truy xuất được:  
    {context}

    Trả lời:
    """)

# Khởi tạo embedding và vectorstore toàn cục (có thể dùng lại)
load_env()
embedding, vectorstore = get_embedding_and_vectorstore()

@tool
def rag_tool(query: str) -> str:
    """
    Trả lời câu hỏi dựa trên vectorstore đã khởi tạo từ ChromaDB.
    Vectorstore phải được gán trước khi gọi tool này.
    """
    global vectorstore
    if vectorstore is None:
        raise ValueError("Vectorstore chưa được gán. Vui lòng khởi tạo trong Agent và gán vào rag_tool_module.vectorstore.")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain.invoke(query)["result"]

if __name__ == "__main__":
    test_query = "Gợi ý nhà hàng 3 sao gần biển"
    print("⚙️ Thực thi rag_tool với query:\n", test_query)
    try:
        result = rag_tool(test_query)
        print("🤖 Kết quả:\n", result)
    except Exception as e:
        print("❌ Lỗi khi chạy rag_tool:", e)

