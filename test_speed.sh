python eval_speed.py \
 --num_workers 1 --batch_size 1  --dataset unc  --model 'ViT-B/16'  --imsize 384 \
 --max_query_len 77 --ce --ce_keep 1.0 \
 --data_root image_data --split_root split_data \
 --eval_model outputs/referit/osvg_clip_vitb16_512_g2_b32_c0.7/best_checkpoint.pth  \
 --eval_set val --device cuda

python eval_speed.py \
 --num_workers 1 --batch_size 1  --dataset unc  --model 'ViT-B/16'  --imsize 384 \
 --max_query_len 77 --ce --ce_keep 0.9 \
 --data_root image_data --split_root split_data \
 --eval_model outputs/referit/osvg_clip_vitb16_512_g2_b32_c0.7/best_checkpoint.pth  \
 --eval_set val --device cuda

python eval_speed.py \
 --num_workers 1 --batch_size 1  --dataset unc  --model 'ViT-B/16'  --imsize 384 \
 --max_query_len 77 --ce --ce_keep 0.8 \
 --data_root image_data --split_root split_data \
 --eval_model outputs/referit/osvg_clip_vitb16_512_g2_b32_c0.7/best_checkpoint.pth  \
 --eval_set val --device cuda

python eval_speed.py \
 --num_workers 1 --batch_size 1  --dataset unc  --model 'ViT-B/16'  --imsize 384 \
 --max_query_len 77 --ce --ce_keep 0.7 \
 --data_root image_data --split_root split_data \
 --eval_model outputs/referit/osvg_clip_vitb16_512_g2_b32_c0.7/best_checkpoint.pth  \
 --eval_set val --device cuda

python eval_speed.py \
 --num_workers 1 --batch_size 1  --dataset unc  --model 'ViT-B/16'  --imsize 384 \
 --max_query_len 77 --ce --ce_keep 0.6 \
 --data_root image_data --split_root split_data \
 --eval_model outputs/referit/osvg_clip_vitb16_512_g2_b32_c0.7/best_checkpoint.pth  \
 --eval_set val --device cuda

python eval_speed.py \
 --num_workers 1 --batch_size 1  --dataset unc  --model 'ViT-B/16'  --imsize 384 \
 --max_query_len 77 --ce --ce_keep 0.5 \
 --data_root image_data --split_root split_data \
 --eval_model outputs/referit/osvg_clip_vitb16_512_g2_b32_c0.7/best_checkpoint.pth  \
 --eval_set val --device cuda
