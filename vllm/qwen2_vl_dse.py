from typing import Iterable, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch import nn
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.attention import AttentionMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.inputs import INPUT_REGISTRY
from vllm.model_executor.models.qwen2_vl import (
  image_input_mapper_for_qwen2_vl,
  video_input_mapper_for_qwen2_vl,
  get_max_qwen2_vl_image_tokens,
  get_max_qwen2_vl_video_tokens,
  dummy_data_for_qwen2_vl,
  input_processor_for_qwen2_vl,
  Qwen2VLForConditionalGeneration
)

def get_query_prompt(query: str) -> Tuple[str, Image.Image]:
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: {query}<|im_end|>\n<|endoftext|>"
    return prompt, Image.new('RGB', (56, 56))

def get_document_prompt(document: Image.Image) -> Tuple[str, Image.Image]:
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>"
    return prompt, document

@MULTIMODAL_REGISTRY.register_image_input_mapper(image_input_mapper_for_qwen2_vl)
@MULTIMODAL_REGISTRY.register_input_mapper("video",video_input_mapper_for_qwen2_vl)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_qwen2_vl_image_tokens)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("video", get_max_qwen2_vl_video_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_qwen2_vl)
@INPUT_REGISTRY.register_input_processor(input_processor_for_qwen2_vl)
class Qwen2VLForEmbeddingGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    A model that uses Qwen2VL with additional embedding functionalities.

    This class encapsulates the Qwen2VLForConditionalGeneration and provides an interface for
    embedding operations and customized pooling functions.

    Attributes:
        model: An instance of Gemma2Model used for forward operations.
        _pooler: An instance of Pooler used for pooling operations.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model = Qwen2VLForConditionalGeneration(**kwargs)
        self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.model.load_weights(weights)
