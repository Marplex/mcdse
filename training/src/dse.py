import torch
from torch import nn
from transformers import Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from arguments import ModelArguments

class DseQwen2(nn.Module):
    def __init__(self, model_args: ModelArguments):
        super().__init__()
        self.encoder = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.encoder.padding_side = "left"

        if model_args.lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path)
            self.encoder = PeftModel.from_pretrained(
                self.encoder,
                model_args.lora_name_or_path,
                torch_dtype=torch.bfloat16,
                is_trainable=True
            )
        elif model_args.lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=model_args.lora_target_modules,
                init_lora_weights="gaussian",
                task_type="FEATURE_EXTRACTION",
                bias="none",
                use_dora=True,
                inference_mode=False
            )

            self.encoder = get_peft_model(self.encoder, lora_config)

        self.encoder.print_trainable_parameters()
        

    def forward(self, **kwargs) -> torch.Tensor:
        cache_position = torch.arange(0, kwargs['input_ids'].shape[0], device=kwargs['input_ids'].device)
        inputs = self.encoder.prepare_inputs_for_generation(**kwargs, use_cache=True, cache_position=cache_position)

        outputs = self.encoder(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )
        
        return self._pooling(outputs.hidden_states[-1])
    
    def _pooling(self, last_hidden_state):
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps