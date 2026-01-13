# Two-Stage Training

1. **Stage 1 – Teacher Reinforcement Learning Policy:**
   A privileged teacher policy is trained using full state information. This allows fast and efficient learning in simulation with access to all relevant data.

2. **Stage 2 – Vision-Based Student Policy:**
   Knowledge from the teacher policy is distilled into a student policy that uses only camera observations (and optionally robot proprioception). This step bridges the gap toward real-world deployment, where full state information is unavailable.


## Training

- Train the RL teacher policy:
```bash
python3 grasp_train.py --stage=rl
python3 grasp_train.py --stage=rl --num_envs=1
```
- Train the student BC policy (requires a trained RL teacher):
```bash
python3 grasp_train.py --stage=bc
```

## Evaluation
- Evaluate a trained RL policy:
```bash
python3 grasp_eval.py --stage=rl
```
- Evaluate a trained BC policy:
```bash
python3 grasp_eval.py --stage=bc
```
- Record evaluation videos:
```bash
python3 grasp_eval.py --stage=bc --record
```

## Camera Setups
The environment supports configurable camera layouts, allowing different perception setups to be selected at initialization time.
Currently, two camera setups are available:
* **default** – matches the real robot setup (without the front tool camera): one static top-down scene camera plus two eye-in-hand cameras moving with the end-effector
* **dual_scene** – legacy configuration: two static scene cameras forming a stereo view from the front
