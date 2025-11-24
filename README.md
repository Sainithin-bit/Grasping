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
