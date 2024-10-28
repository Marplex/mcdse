import os
from typing import Optional
import torch
from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist

class DseTrainer(Trainer):
    def __init__(self, loss_func, model_output_dir, *args, **kwargs):
        super(DseTrainer, self).__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.model_output_dir = model_output_dir
        self.world_size = dist.get_world_size()
        self.process_rank = dist.get_rank()

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.encoder.save_pretrained(self.model_output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.model_output_dir)

    def _dist_gather_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors)

    def compute_loss(self, model, inputs, num_items_in_batch):
        query, passage = inputs

        query_reps = model(**query)
        passage_reps = model(**passage)

        query_reps = self._dist_gather_tensor(query_reps)
        passage_reps = self._dist_gather_tensor(passage_reps)

        return self.loss_func(query_reps, passage_reps) * self.world_size

    def training_step(self, *args):
        return super(DseTrainer, self).training_step(*args) / self.world_size
