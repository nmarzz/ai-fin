#!/bin/bash

# Trains each model ()['ppo','ddpg','a2c','td3','sac']) for the specified number of training steps
TRAINSTEPS=$1
STARTDATE=$2

for MODEL in ppo ddpg a2c td3 sac
do
  python train_on_yahoo.py\
  --model $MODEL\
  --train-steps $TRAINSTEPS\
  --start-date $STARTDATE
done
