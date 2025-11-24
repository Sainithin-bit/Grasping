# Robotic Grasping with RL and Behavior Cloning

This repository contains code for training and evaluating robotic grasping policies using **Reinforcement Learning (RL)** and **Behavior Cloning (BC)**. The goal is to develop a robotic agent capable of approaching objects and grasping them effectively using a combination of RL and BC methods.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Training](#training)
  - [Train RL Policy](#train-rl-policy)
  - [Train BC Policy](#train-bc-policy)
- [Evaluation](#evaluation)
  - [Evaluate RL Policy](#evaluate-rl-policy)
  - [Evaluate BC Policy](#evaluate-bc-policy)
  - [Evaluation with Video Recording](#evaluation-with-video-recording)
- [Notes](#notes)
- [References](#references)

---

## Environment Setup

We recommend using **conda** to manage dependencies.  

```bash
# Create the environment from the provided YAML file
conda env create -f st.yml

# Activate the environment
conda activate <env_name>
```

## 🎓 Training

### 1. Train RL Policy

This step trains the **Reinforcement Learning (RL)** policy from scratch.

```bash
python grasp_train.py --stage=rl

```

Behavior Cloning (BC) training requires a pre-trained RL policy as a reference for demonstration data.

```bash
python grasp_train.py --stage=bc

```


## Evaluation

The evaluation scripts test the performance of the trained policies.

```bash
python grasp_eval.py --stage=bc --record
```
## Notes and Customization

**Prerequisites**: Ensure that the RL policy is trained before starting BC training.

**Hyperparameters**: Adjust relevant hyperparameters in grasp_train.py and grasp_eval.py according to your environment and requirements.

**Customization**: Evaluation scripts can be modified to test on different objects or scenarios.

**Debugging**: Recorded videos are useful for debugging and visualizing agent behavior.