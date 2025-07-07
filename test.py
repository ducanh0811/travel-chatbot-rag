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
                tags_list = item.get("tags", [])
                tags_str = ", ".join(tags_list)
                duration = item.get("duration_suggested_min", "Không rõ")
                content = (
                    f"{name} là một {data_type} nằm ở phường {ward}, quận {district}. "
                    f"Mô tả: {desc} "
                    f"Mở cửa từ {open_time} đến {close_time}. "
                    f"Gợi ý thời gian tham quan: khoảng {duration} phút. "
                    f"Từ khóa liên quan: {', '.join(tags_str)}."
                )

                doc = Document(
                    page_content=content,
                    metadata={
                        "type": data_type,
                        "name": name,
                        "district":str(district) if district else None ,
                        "ward": str(ward) if ward else None,
                        "tags": tags_str  # ✅ Lưu đúng dạng list
                    }
                )
                all_docs.append(doc)
    return all_docs

# ====== Hàm lọc theo metadata (ward, district, tag) ======
def get_retriever_with_filters(vectorstore, district=None, ward=None, tag=None):
    filters = {}

    if district:
        filters["district"] = {"$eq": district}
    if ward:
        filters["ward"] = {"$eq": ward}
    if tag:
        filters["tags"] = {"$in": [tag.lower()]}

    if filters:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "filter": {"where": filters}  # ✅ Bọc trong where
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

# ====== Trích xuất tag từ câu hỏi người dùng ======
def extract_tags_from_input(user_input, all_tags):
    matched_tags = []
    for tag in all_tags:
        if tag.lower() in user_input.lower():
            matched_tags.append(tag.lower())
    return matched_tags

def get_all_tags(documents):
    tag_set = set()
    for doc in documents:
        tags = doc.metadata.get("tags", [])
        if isinstance(tags, list):
            tag_set.update([t.lower() for t in tags])
    return list(tag_set)

# ====== Truy vấn với LangChain ======
def build_chatbot(vectorstore, district=None, ward=None, tag=None):
    retriever = get_retriever_with_filters(vectorstore, district, ward, tag)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        temperature=0.7
    )

    QA_PROMPT = PromptTemplate.from_template("""
    Bạn là một hướng dẫn viên du lịch thông minh. Trả lời câu hỏi của người dùng bằng tiếng Việt, dựa trên thông tin từ dữ liệu địa điểm bên dưới.

    Nếu người dùng cần gợi ý, hãy đưa ra 5 địa điểm phù hợp nhất. Mỗi gợi ý trình bày theo mẫu:

    Tên: (Tên địa điểm)  
    Loại: (Loại địa điểm: quán ăn, cafe, khách sạn...)  
    Khu vực: (Phường, Quận)  
    Mô tả: (Mô tả ngắn gọn, hấp dẫn, nổi bật)  
    ---

    Nếu người dùng yêu cầu chi tiết hơn, bạn có thể mô tả sâu hơn về các dịch vụ, giờ mở cửa, phù hợp với nhóm nào, v.v.

    Câu hỏi của người dùng: {question}  
    Dữ liệu địa điểm truy xuất được:  
    {context}

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
    all_tags = get_all_tags(documents)

    print(f"📄 Đã tải {len(documents)} địa điểm.")
    print("💾 Đang tạo vector store...")
    vectorstore = create_vector_store(documents)

    print("🤖 Khởi động chatbot...")
    print("✅ Chatbot đã sẵn sàng! Gõ 'exit' để thoát.")
    history = []

    while True:
        user_input = input("💬 Bạn: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        if user_input in ["chi tiết hơn", "xem thêm", "nói rõ hơn", "tell me more"]:
            if history:
                user_input = f"Cho tôi biết chi tiết hơn về {history[-1]}"
            else:
                print("🤖 Bot: Bạn muốn biết chi tiết về gì?")
                continue
        else:
            history.append(user_input)

        matched_tags = extract_tags_from_input(user_input, all_tags)
        tag_filter = matched_tags[0] if matched_tags else None

        chatbot = build_chatbot(vectorstore, tag=tag_filter)
        result = chatbot.invoke({"query": user_input})
        print("🤖 Bot:", result["result"])
