# mcdse

mcdse-2b-v1 is a new experimental multilingual model for OCR-free document retrieval.

This model allows you to embed page/slide screenshots and query them using natural language. Tables, graphs, charts, schemas, images and text are "automagically" encoded for you into a single embedding vector. No need to worry about OCR, document layout analysis, reading order detection, table/formula extraction...

This repo adds vLLM support for generating embeddings using mcdse-2b-v1

### Download mcdse-2b-v1 for local inference
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="marco/mcdse-2b-v1", local_dir="/path/to/model/mcdse-2b-v1")
```

### Edit config.json
Replace `Qwen2VLForConditionalGeneration` with `Qwen2VLForEmbeddingGeneration`
```bash
sed -i -e 's/Qwen2VLForConditionalGeneration/Qwen2VLForEmbeddingGeneration/g' /path/to/model/mcdse-2b-v1/config.json
```

### Open `vllm/main.py` for usage instructions