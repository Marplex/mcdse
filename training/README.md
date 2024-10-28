# Training

Training using LoRA on French
```bash
python -m torch.distributed.launch --nproc_per_node=8 ./src/train.py --json_file ./configs/mcdse_tensorboard_fr.json
```