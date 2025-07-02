import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
# ====== Load API Key ======
load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]

# ====== Load & Gộp dữ liệu từ 4 file JSON ======
def load_all_data():
    file_paths = {
        "hotel": "data/hotel.json",
        "restaurant": "data/restaurant.json",
        "destination": "data/destination.json",
        "cafe": "data/cafe.json"
    }

    all_docs = []
    for data_type, path in file_paths.items():
        if not os.path.exists(path):
            print(f"⚠️ Không tìm thấy file: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

            for item in items:
                name = item.get("name", "Không rõ")
                desc = item.get("description", "")
                district = item.get("district", "")
                ward = item.get("ward", "")
                if data_type == "hotel":
                    open_time = item.get("checkin_time", "Không rõ")
                    close_time = item.get("checkout_time", "Không rõ")
                else:
                    open_time = item.get("open_time", "")
                    close_time = item.get("close_time", "")
                tags = ", ".join(item.get("tags", []))
                duration = item.get("duration_suggested_min", "Không rõ")

                content = (
                    f"{name} là một {data_type} nằm ở phường {ward}, quận {district}. "
                    f"Mô tả: {desc} "
                    f"Mở cửa từ {open_time} đến {close_time}. "
                    f"Gợi ý thời gian tham quan: khoảng {duration} phút. "
                    f"Từ khóa liên quan: {tags}."
                )

                doc = Document(
                    page_content=content,
                    metadata={
                        "type": data_type,
                        "name": name,
                        "district": str(item.get("district", "")) if item.get("district") else None,
                        "ward": str(item.get("ward", "")) if item.get("ward") else None,
                        "tags": ", ".join(item.get("tags", []))
                    }
                )
                all_docs.append(doc)
    return all_docs
#======Hàm lọc theo metadata(ward, district) =====
def get_retriever_with_filters(vectorstore, district=None, ward=None):
    filters = {}

    if district:
        filters["district"] = district
    if ward:
        filters["ward"] = ward
    if filters:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "filter": filters
            }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )
    return retriever
# ====== Tạo Embedding và lưu vào ChromaDB ======
def create_vector_store(documents):
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory="chromadb"
    )
    vectorstore.persist()
    return vectorstore

# ====== Truy vấn với LangChain ======
def build_chatbot(vectorstore, district=None, ward=None):
    retriever = get_retriever_with_filters(vectorstore, district, ward)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        temperature=0.7  # Gợi ý để tạo câu trả lời phong phú hơn
    )

    # chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     return_source_documents=False
    # )
    QA_PROMPT = PromptTemplate.from_template("""
    Bạn là một hướng dẫn viên du lịch thông minh. Trả lời câu hỏi của người dùng bằng tiếng Việt, dựa trên thông tin được cung cấp từ dữ liệu truy xuất.

    ❗ Nếu người dùng hỏi gợi ý, hãy trình bày theo format sau:

    Tên: ...
    Loại: ...
    Khu vực: ...
    ❗ Nếu người dùng nói "chi tiết hơn" hoặc "xem thêm", hãy mở rộng mô tả và đưa thêm các điểm nổi bật.

    Câu hỏi: {question}
    Thông tin hỗ trợ: {context}
    Trả lời:
    """)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )    
    return chain



# ====== MAIN ======
if __name__ == "__main__":
    print("🔄 Đang tải dữ liệu...")
    documents = load_all_data()

    print(f"📄 Đã tải {len(documents)} địa điểm.")
    print("💾 Đang tạo vector store...")
    vectorstore = create_vector_store(documents)

    print("🤖 Khởi động chatbot...")
    chatbot = build_chatbot(vectorstore)

    print("✅ Chatbot đã sẵn sàng! Gõ 'exit' để thoát.")
    history = []
    while True:
        user_input = input("💬 Bạn: ").strip()
        if user_input in ["chi tiết hơn", "xem thêm", "nói rõ hơn", "tell me more"]:
                if history:
                    user_input = f"Cho tôi biết chi tiết hơn về {history[-1]}"
                else:
                    print("🤖 Bot: Bạn muốn biết chi tiết về gì?")
                    continue
        else:
                history.append(user_input)

        result = chatbot.invoke({"query": user_input})
        print("🤖 Bot:", result["result"])