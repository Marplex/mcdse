![](art/cover_wide.png)

**mcdse-2b-v1** is a new experimental multilingual model for OCR-free document retrieval.

This model allows you to embed page/slide screenshots and query them using natural language. Tables, graphs, charts, schemas, images and text are "automagically" encoded for you into a single embedding vector. No need to worry about OCR, document layout analysis, reading order detection, table/formula extraction...

- **Understands 🇮🇹 Italian, 🇪🇸 Spanish, 🇬🇧 English, 🇫🇷 French and 🇩🇪 German**

- **Matryoshka Representation Learning:** shrink embeddings from 1536 to 256 dimensions while maintaining 95% of the quality. A 6x reduction with negligible impact on performance!

- **Top-tier Binarization**: 768-dimensional binary vectors retain 99% retrieval quality of the original 1536-dimensional float vectors. With binary vectors, you can encode **100 million multilingual pages in just 10GB**.

- **Fast vLLM inference:** run inference on vLLM and efficiently serve embeddings at scale, production ready.

For more information about this model or how it was trained, visit the [announcement blogpost](https://huggingface.co/blog/marco/announcing-mcdse-2b-v1).

## Evaluations
Given the scarcity of publicly available datasets for multilingual document image retrieval, the model has been evaluated using a custom-built dataset. This eval dataset was specifically designed to benchmark the model's performance across various languages.

### NDCG@5 (float)
|                     | Average    | English    | Italian    | Spanish    | French     | German     |
|---------------------|------------|------------|------------|------------|------------|------------|
| **1536 dimensions** |            |            |            |            |            |            |
| dse-qwen2-2b-mrl-v1 |       79.5 |       79.2 |       80.2 |       77.9 |       80.6 |       79.6 |
| mcdse-2b-v1         |   **82.2** |   **80.8** |   **81.2** |   **80.7** |   **84.5** |   **83.8** |
|                     | **+3.28%** | **+1.98%** | **+1.23%** | **+3.47%** | **+4.62%** | **+5.01%** |
| **1024 dimensions** |            |            |            |            |            |            |
| dse-qwen2-2b-mrl-v1 |       78.3 |       78.8 |       78.5 |       76.5 |         80 |       77.5 |
| mcdse-2b-v1         |   **81.7** |     **80** |   **80.2** |   **80.1** |     **84** |   **84.3** |
|                     | **+4.23%** | **+1.75%** | **+2.12%** | **+4.49%** | **+4.76%** | **+8.07%** |
| **768 dimensions**  |            |            |            |            |            |            |
| dse-qwen2-2b-mrl-v1 |       77.8 |       78.4 |       78.3 |       75.6 |       80.8 |       75.9 |
| mcdse-2b-v1         |   **81.1** |   **79.6** |   **79.9** |   **79.2** |   **83.3** |   **83.3** |
|                     | **+4.02%** | **+1.51%** | **+2.00%** | **+4.55%** | **+3.00%** | **+8.88%** |
| **512 dimensions**  |            |            |            |            |            |            |
| dse-qwen2-2b-mrl-v1 |       76.2 |       77.6 |       75.9 |       73.1 |       79.2 |       75.2 |
| mcdse-2b-v1         |   **79.3** |   **78.5** |   **79.1** |   **75.8** |   **81.4** |   **81.7** |
|                     | **+3.91%** | **+1.15%** | **+4.05%** | **+3.56%** | **+2.70%** | **+7.96%** |
| **384 dimensions**  |            |            |            |            |            |            |
| dse-qwen2-2b-mrl-v1 |       75.7 |       76.2 |       75.5 |       74.6 |       78.4 |         74 |
| mcdse-2b-v1         |   **78.8** |   **77.5** |   **78.5** |   **76.1** |   **80.4** |   **81.4** |
|                     | **+3.86%** | **+1.68%** | **+3.82%** | **+1.97%** | **+2.49%** | **+9.09%** |
| **256 dimensions**  |            |            |            |            |            |            |
| dse-qwen2-2b-mrl-v1 |       73.5 |       74.5 |       73.6 |       70.6 |       74.8 |       73.8 |
| mcdse-2b-v1         |   **78.1** |   **78.5** |   **77.6** |   **76.2** |   **80.1** |   **77.9** |
|                     | **+5.89%** | **+5.10%** | **+5.15%** | **+7.35%** | **+6.62%** | **+5.26%** |

### NDCG@5 (binary)
|                     | Average     | English     | Italian     | Spanish     | French      | German      |
|---------------------|-------------|-------------|-------------|-------------|-------------|-------------|
| **1536 dimensions** |             |             |             |             |             |             |
| dse-qwen2-2b-mrl-v1 |        75.0 |        75.8 |        75.4 |        72.4 |        78.1 |        73.2 |
| mcdse-2b-v1         |    **80.6** |    **79.5** |    **76.9** |    **81.9** |    **83.7** |    **80.8** |
|                     |  **+6.93%** |  **+4.65%** |  **+1.95%** | **+11.60%** |  **+6.69%** |  **+9.41%** |
| **1024 dimensions** |             |             |             |             |             |             |
| dse-qwen2-2b-mrl-v1 |        72.2 |        74.8 |          71 |        70.8 |        74.6 |        69.6 |
| mcdse-2b-v1         |    **79.3** |    **78.4** |    **75.4** |    **80.8** |    **82.6** |    **79.5** |
|                     |  **+9.05%** |  **+4.59%** |  **+5.84%** | **+12.38%** |  **+9.69%** | **+12.45%** |
| **768 dimensions**  |             |             |             |             |             |             |
| dse-qwen2-2b-mrl-v1 |        70.1 |        71.7 |        69.3 |        69.8 |        73.7 |        65.9 |
| mcdse-2b-v1         |    **78.8** |    **77.1** |    **75.4** |      **80** |      **83** |    **78.5** |
|                     | **+11.07%** |  **+7.00%** |  **+8.09%** | **+12.75%** | **+11.20%** | **+16.05%** |
| **512 dimensions**  |             |             |             |             |             |             |
| dse-qwen2-2b-mrl-v1 |        66.5 |          70 |        65.4 |        63.7 |        70.2 |          63 |
| mcdse-2b-v1         |    **76.6** |    **74.8** |    **74.2** |    **77.7** |    **80.9** |    **75.3** |
|                     | **+13.21%** |  **+6.42%** | **+11.86%** | **+18.02%** | **+13.23%** | **+16.33%** |
| **384 dimensions**  |             |             |             |             |             |             |
| dse-qwen2-2b-mrl-v1 |        61.1 |        62.7 |        58.5 |        58.6 |        65.1 |        60.8 |
| mcdse-2b-v1         |    **74.3** |    **74.5** |    **71.4** |    **77.2** |    **75.2** |      **73** |
|                     | **+17.67%** | **+15.84%** | **+18.07%** | **+24.09%** | **+13.43%** | **+16.71%** |
| **256 dimensions**  |             |             |             |             |             |             |
| dse-qwen2-2b-mrl-v1 |        54.3 |          59 |        56.5 |        53.6 |          53 |        49.6 |
| mcdse-2b-v1         |    **70.9** |    **72.6** |    **66.4** |    **73.5** |    **72.6** |    **69.2** |
|                     | **+23.31%** | **+18.73%** | **+14.91%** | **+27.07%** | **+27.00%** | **+28.32%** |



## vLLM
This repo implements a new model class `Qwen2VLForEmbeddingGeneration` to support embedding generation with Qwen2VL models.

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