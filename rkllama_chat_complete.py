from llama_index.llms.ollama import Ollama

#  Khởi tạo
ollama = Ollama(model="Qwen2.5-7B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0", base_url="http://127.0.0.1:8080", request_timeout=500)

# Câu hỏi thử nghiệm
question = "Bạn là ai?"

# Câu trả lời
result = ollama.complete(question)
print(result)

