#!/bin/bash
python3 grasp_train.py --stage=rl --num-envs=1 --max-iterations 10
python3 grasp_train.py --stage=bc --num-envs=1 --max-iterations 10 --load-rl-task=aab77449d7f449349addf2c46fdb629a
python3 grasp_eval.py --stage=rl --num-envs=1 --max-iterations 10 --load-rl-task=aab77449d7f449349addf2c46fdb629a
python3 grasp_eval.py --stage=bc --num-envs=1 --max-iterations 10 --load-rl-task=aab77449d7f449349addf2c46fdb629a
