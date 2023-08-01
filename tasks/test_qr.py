import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

validation_prompt = ("fairy in the garden, unreal engine, 4k")

cmd = rf'''CUDA_VISIBLE_DEVICES="1" accelerate launch --num_processes=1 --mixed_precision="fp16" test_text_to_image_control_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="process/laion_qr" --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=30 --checkpointing_steps=5000 --resume_from_checkpoint="latest" \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 --validation_epochs=5 \
  --output_dir="ckpts/sd-laion_qr-control-lora" \
  --control_lora_config="configs/diffusiondb-canny-v2.json" \
  --validation_prompt="{validation_prompt}" --report_to="wandb" '''

os.system(cmd.replace('\\', ' ').replace('\r\n', '\n').replace('\n', ''))
