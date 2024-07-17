# 在没有显示器的机器上请提前先运行，用于启动虚拟服务器: xvfb-run -a -s "-screen 0 1400x900x24" bash
# 程序运行完毕后，请手动关闭该服务器进程: ps aux | grep Xvfb; kill xxxx


source /home/fu/anaconda3/envs/dmcontrol

CUDA_ID=0
DOMAIN_NAME=walker
TASK_NAME=walk
ENCODER_TYPE=pixel
PRE_TRANSFORM_IMAGE_SIZE=100
IMAGE_SIZE=84
AGENT=ctmr_sac
CRITIC_LR=1e-3
ACTOR_LR=1e-3
BATCH_SIZE=128
NUM_TRAIN_STEPS=500000
EVAL_FREQ=2000

SEED = (1, 26, 1024)

for SEED in "${SEED[@]}"; do
  python train.py \
      --eval_freq $eval_freq \
      --cuda_id $CUDA_ID \
      --domain_name $DOMAIN_NAME \
      --task_name $TASK_NAME \
      --encoder_type $ENCODER_TYPE \
      --pre_transform_image_size $PRE_TRANSFORM_IMAGE_SIZE \
      --image_size $IMAGE_SIZE \
      --work_dir ./results/$DOMAIN_NAME-$TASK_NAME \
      --agent $AGENT \
      --seed $SEED \
      --critic_lr $CRITIC_LR \
      --actor_lr $ACTOR_LR \
      --batch_size $BATCH_SIZE \
      --num_train_steps $NUM_TRAIN_STEPS \
  #    --save_tb \
  #    --save_video \
  #    --save_buffer \
  #    --action_repeat 4 \
  #    --encoder_type pixel_ctmr
  done