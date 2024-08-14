srun -p ai4earth --gres=gpu:1 --job-name=show --quotatype=auto python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env  show_result.py \
 --ce --ce_keep 0.7 --min_keep 0.0 --max_keep 0.5 --model 'ViT-L/14' --imsize 384 \
 --eval_model outputs/unc/osvg_clip_vitl_384_g4_b16_c0.7/best_checkpoint.pth

# srun -p ai4earth --gres=gpu:1 --job-name=show --quotatype=auto python  show_result.py \
#  --ce --ce_keep 0.8 --min_keep 0.0 --max_keep 0.343 \
#  --eval_model outputs/referit/osvg_clip_vitb16_384_g2_b64_c0.8/best_checkpoint.pth

# srun -p ai4earth --gres=gpu:1 --job-name=show --quotatype=auto python  show_result.py \
#  --ce --ce_keep 0.8 --min_keep 0.343 --max_keep 0.512 \
#  --eval_model outputs/referit/osvg_clip_vitb16_384_g2_b64_c0.8/best_checkpoint.pth

# srun -p ai4earth --gres=gpu:1 --job-name=show --quotatype=auto python show_result.py \
#  --ce --ce_keep 0.8 --min_keep 0.512 --max_keep 0.729 \
#  --eval_model outputs/referit/osvg_clip_vitb16_384_g2_b64_c0.8/best_checkpoint.pth

# srun -p ai4earth --gres=gpu:1 --job-name=show --quotatype=auto python show_result.py \
#  --ce --ce_keep 0.8 --min_keep 0.729 --max_keep 1.0 \
#  --eval_model outputs/referit/osvg_clip_vitb16_384_g2_b64_c0.8/best_checkpoint.pth

