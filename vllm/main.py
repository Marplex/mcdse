from qwen2_vl_dse import Qwen2VLForEmbeddingGeneration, get_query_prompt, get_document_prompt
from vllm import ModelRegistry, LLM
from PIL import Image

ModelRegistry.register_model("Qwen2VLForEmbeddingGeneration", Qwen2VLForEmbeddingGeneration)

llm = LLM(
    model="/path/to/model/mcdse-2b-v1",
    limit_mm_per_prompt={
        "image": 1
    }
)

# Encode queries
query_prompt, image = get_query_prompt("Quali erano le passività totali al 31 dicembre 2017?")
outputs = llm.encode({"prompt": query_prompt, "multi_modal_data": {"image": [image]}})
outputs[0].outputs.embedding #1536 dimensional embedding

# Encode documents
dummy_document_image = Image.new('RGB', (256, 256))
document_prompt, image = get_document_prompt(dummy_document_image)
outputs = llm.encode({"prompt": document_prompt, "multi_modal_data": {"image": [image]}})
outputs[0].outputs.embedding #1536 dimensional embedding
