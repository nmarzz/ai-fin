#!/bin/bash

# Trains each model ()['ppo','ddpg','a2c','td3','sac']) for the specified number of training steps
trainsteps = $1

for model in ppo ddpg a2c td3 sac
do
  python train_on_yahoo.py\
   --model $model
   --train-steps trainsteps
done
