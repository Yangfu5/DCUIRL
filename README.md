# DCUIRL: Dynamic Contrastive RepresentationLearning for Inverse Reinforcement Learning

This repository contains the code for DCUIRL.

## Installation 

All of the dependencies are in the `conda_env.yml` file.
They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Instructions
To train an DCUIRL agent on the `finger spin` task,  please use `script/run.sh` 
from the root of this directory. One example is as follows, 
and you can modify it to try different environments / hyperparamters.
```

CUDA_ID=0
DOMAIN_NAME=finger
TASK_NAME=spin

ENCODER_TYPE=pixel
PRE_TRANSFORM_IMAGE_SIZE=100
IMAGE_SIZE=84
AGENT=ctmr_sac
CRITIC_LR=1e-3
ACTOR_LR=1e-3
BATCH_SIZE=128
NUM_TRAIN_STEPS=500000
EVAL_FREQ=2000

SEED=(1 26 1024)

for SEED in "${SEED[@]}"; do
  python train.py \
      --eval_freq $EVAL_FREQ \
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
      --is_irl
  #    --save_tb \
  #    --save_video \
  #    --save_buffer \
  #    --action_repeat 4 \
  done
```
