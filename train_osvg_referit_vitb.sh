EPOCHS=90
GPUS=2
UNM_WORKERS=2
BATCH_SIZE=64

LR=1e-4
LR_VISUAL=1e-5
LR_TEXT=1e-5
LR_DROP=60
BBOX_W=5
GIOU_W=2

MODEL='ViT-B/16' # ViT-B/16  ViT-B/32 ViT-L/14
IMAGE_SIZE=256

DATA_NAME=referit # unc unc+ gref gref_umd referit
OUTPUT=outputs/${DATA_NAME}/osvg_clip_vitb16_${IMAGE_SIZE}_g${GPUS}_b${BATCH_SIZE}

srun -p ai4earth --gres=gpu:$GPUS --job-name=${DATA_NAME} --quotatype=reserved \
 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 28601 --use_env train_osvg.py \
 --num_workers $UNM_WORKERS --epochs $EPOCHS --lr_drop $LR_DROP --batch_size $BATCH_SIZE --lr $LR --lr_visu $LR_VISUAL --lr_text $LR_TEXT \
 --lr_scheduler step --bbox_w $BBOX_W --giou_w $GIOU_W --aug_crop --aug_scale --aug_translate \
 --model $MODEL --imsize $IMAGE_SIZE --max_query_len 50 \
 --sup_type full --dataset $DATA_NAME --data_root image_data --split_root split_data \
 --output_dir ${OUTPUT} --device cuda --resume ${OUTPUT}/checkpoint.pth

srun -p ai4earth --gres=gpu:$GPUS --job-name=$DATA_NAME --quotatype=spot python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 28888 --use_env eval_osvg.py \
 --num_workers $UNM_WORKERS --batch_size 1  --dataset $DATA_NAME  --model $MODEL  --imsize $IMAGE_SIZE --max_query_len 77 \
 --data_root image_data --split_root split_data --eval_model ${OUTPUT}/best_checkpoint.pth  \
 --eval_set val --device cuda --output_dir ${OUTPUT}

srun -p ai4earth --gres=gpu:$GPUS --job-name=$DATA_NAME --quotatype=spot python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 28888 --use_env eval_osvg.py \
 --num_workers $UNM_WORKERS --batch_size $BATCH_SIZE  --dataset $DATA_NAME  --model $MODEL  --imsize $IMAGE_SIZE --max_query_len 77 \
 --data_root image_data --split_root split_data --eval_model ${OUTPUT}/best_checkpoint.pth  \
 --eval_set test --device cuda --output_dir ${OUTPUT}