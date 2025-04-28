# Import các thư viện cần thiết
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Thiết lập cấu hình
# Sử dụng mô hình nhúng văn bản từ HuggingFace
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Sử dụng mô hình ngôn ngữ Ollama chạy cục bộ
# Thêm system prompt để chỉ định sử dụng tiếng Việt
Settings.llm = Ollama(
    model="Qwen2.5-7B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0", 
    base_url="http://127.0.0.1:8080", 
    request_timeout=500,
    system_prompt="Bạn là trợ lý AI hữu ích. Hãy luôn trả lời bằng tiếng Việt. Khi suy luận và đưa ra kết quả, hãy sử dụng tiếng Việt."
)

# Định nghĩa các công cụ tính thuế GTGT

def tinh_gia_co_thue(gia_chua_thue: float, thue_suat: float = 8.0) -> float:
    """
    Tính giá có thuế từ giá chưa thuế
    
    Tham số:
        gia_chua_thue: Giá sản phẩm chưa bao gồm thuế GTGT
        thue_suat: Thuế suất GTGT (mặc định là 8%)
        
    Trả về:
        Giá sản phẩm đã bao gồm thuế GTGT
    """
    print("Đang tính giá có thuế từ giá chưa thuế...")
    he_so = 1 + (thue_suat / 100)
    gia_co_thue = gia_chua_thue * he_so
    return gia_co_thue

# Tạo công cụ từ hàm tính giá có thuế
tinh_gia_co_thue_tool = FunctionTool.from_defaults(
    fn=tinh_gia_co_thue,
    name="tinh_gia_co_thue",
    description="Tính giá có thuế GTGT từ giá chưa thuế. Ví dụ: giá chưa thuế 100,000đ với thuế suất 8% sẽ có giá sau thuế là 108,000đ"
)

def tinh_gia_chua_thue(gia_co_thue: float, thue_suat: float = 8.0) -> float:
    """
    Tính giá chưa thuế từ giá có thuế
    
    Tham số:
        gia_co_thue: Giá sản phẩm đã bao gồm thuế GTGT
        thue_suat: Thuế suất GTGT (mặc định là 8%)
        
    Trả về:
        Giá sản phẩm chưa bao gồm thuế GTGT
    """
    print("Đang tính giá chưa thuế từ giá có thuế...")
    he_so = 1 + (thue_suat / 100)
    gia_chua_thue = gia_co_thue / he_so
    return gia_chua_thue

# Tạo công cụ từ hàm tính giá chưa thuế
tinh_gia_chua_thue_tool = FunctionTool.from_defaults(
    fn=tinh_gia_chua_thue,
    name="tinh_gia_chua_thue",
    description="Tính giá chưa thuế GTGT từ giá có thuế. Ví dụ: giá có thuế 108,000đ với thuế suất 8% sẽ có giá chưa thuế là 100,000đ"
)

# Tạo dữ liệu mẫu về thuế GTGT để đưa vào hệ thống RAG
import tempfile
import os

# Tạo thư mục tạm thời để lưu dữ liệu
temp_dir = tempfile.mkdtemp()
tax_info_path = os.path.join(temp_dir, "thue_gtgt_info.txt")

# Tạo file văn bản chứa thông tin về thuế GTGT
with open(tax_info_path, "w", encoding="utf-8") as f:
    f.write("""
# Thông tin về thuế GTGT (VAT) ở Việt Nam

Thuế giá trị gia tăng (GTGT) hay còn gọi là thuế VAT (Value Added Tax) là một loại thuế gián thu đánh vào giá trị tăng thêm của hàng hóa, dịch vụ.

## Các mức thuế suất GTGT phổ biến ở Việt Nam:
1. Thuế suất 0%: Áp dụng cho hàng hóa, dịch vụ xuất khẩu
2. Thuế suất 5%: Áp dụng cho các mặt hàng thiết yếu như nước sạch, dụng cụ giáo dục, sách báo, dược phẩm, thiết bị y tế
3. Thuế suất 8%: Áp dụng cho hầu hết các mặt hàng tiêu dùng và dịch vụ từ ngày 01/02/2022 đến hết ngày 31/12/2023 (giảm từ mức 10%)
4. Thuế suất 10%: Áp dụng cho hầu hết các mặt hàng tiêu dùng và dịch vụ (mức chuẩn)

## Cách tính thuế GTGT:
1. Tính từ giá chưa thuế sang giá có thuế:
   Giá có thuế = Giá chưa thuế × (1 + Thuế suất)
   Ví dụ: Sản phẩm giá 100,000đ chưa thuế, thuế suất 8%:
   Giá có thuế = 100,000 × 1.08 = 108,000đ

2. Tính từ giá có thuế sang giá chưa thuế:
   Giá chưa thuế = Giá có thuế ÷ (1 + Thuế suất)
   Ví dụ: Sản phẩm giá 108,000đ đã bao gồm thuế, thuế suất 8%:
   Giá chưa thuế = 108,000 ÷ 1.08 = 100,000đ

3. Tính riêng phần thuế GTGT:
   Tiền thuế GTGT = Giá chưa thuế × Thuế suất
   hoặc
   Tiền thuế GTGT = Giá có thuế - Giá chưa thuế
    """)

# Thiết lập hệ thống RAG (Retrieval-Augmented Generation) cho thông tin thuế
# Đọc dữ liệu từ thư mục tạm
documents = SimpleDirectoryReader(temp_dir).load_data()
# Tạo chỉ mục vector từ các tài liệu
index = VectorStoreIndex.from_documents(documents)
# Tạo công cụ truy vấn từ chỉ mục
query_engine = index.as_query_engine()

# Tạo công cụ từ hệ thống RAG để truy vấn thông tin về thuế GTGT
thue_gtgt_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="thong_tin_thue_gtgt",
    description="Công cụ RAG chứa thông tin về thuế GTGT ở Việt Nam, các mức thuế suất và cách tính thuế."
)

# Tùy chỉnh hướng dẫn cho agent để đảm bảo sử dụng tiếng Việt
CUSTOM_PROMPT = """Bạn là một trợ lý AI hữu ích chuyên về tính toán thuế GTGT ở Việt Nam.
Hãy luôn suy nghĩ và trả lời bằng tiếng Việt.
Khi cần tính toán, hãy sử dụng các công cụ có sẵn.
Đảm bảo kết quả cuối cùng được trình bày rõ ràng, đầy đủ và bằng tiếng Việt.
"""

# Tạo agent với các công cụ tính thuế GTGT và prompt tùy chỉnh
agent = ReActAgent.from_tools(
    [tinh_gia_co_thue_tool, tinh_gia_chua_thue_tool, thue_gtgt_tool], 
    verbose=True,  # Hiển thị quá trình suy luận
    max_iterations=30,  # Giới hạn số lần lặp tối đa
    system_prompt=CUSTOM_PROMPT  # Sử dụng prompt tùy chỉnh để đảm bảo agent sử dụng tiếng Việt
)

# Ví dụ sử dụng agent để tính thuế GTGT
response = agent.chat("Nếu sản phẩm có giá 500,000đ chưa bao gồm thuế, thì giá sau khi tính thuế GTGT 8% là bao nhiêu?")

print(response)
