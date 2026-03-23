# Bias Mitigation in Large Language Models using Reinforcement Learning

## 📌 Overview
This project investigates the use of reinforcement learning to reduce bias in small instruction-tuned Large Language Models (LLMs) under resource-constrained settings.

A Qwen 0.5B Instruct model is fine-tuned using a Group Relative Policy Optimization (GRPO) framework, with a reward function designed to penalize biased outputs while maintaining response quality.

---

## 🎯 Objectives
- Reduce biased and stereotypical responses in LLMs  
- Explore efficient fine-tuning using LoRA and quantization  
- Study trade-offs between bias mitigation and language quality  
- Compare evaluation methods (benchmark, human, LLM-based)

---

## 🧠 Methodology

### Model
- Base Model: `Qwen/Qwen2-0.5B-Instruct`  
- Fine-tuning: LoRA (via PEFT)  
- Quantization: 4-bit NF4 (BitsAndBytes)  

### Training Framework
- Algorithm: Group Relative Policy Optimization (GRPO)  
- Objective: REINFORCE-style loss with KL regularization  
- Multi-sample generation (K responses per prompt)  

### Reward Function
Implemented in `src/reward/bias_reward.py`, combining:
- **Bias score** (DeBERTa, trained on CrowS-Pairs)  
- **Style reward** (fluency and readability)  
- **Length reward** (discourages trivial responses)  
- **Repetition penalties**

---

```
## 📂 Repository Structure
bias-mitigation-rl/
│── DeBERTaV3/ToxiGen/ # Pretrained bias scoring model
│── analysis/ # Analysis scripts / outputs
│── data/
│ ├── prompts/ # Training prompts
│ ├── model_comparisons_* # Evaluation outputs for different configs
│ └── plots/ # Generated plots
│
│── notebooks/
│ └── 01_baseline_bias_eval.ipynb
│
│── src/
│ ├── data_prep/ # Data preprocessing
│ ├── eval/ # Evaluation scripts
│ ├── reward/ # Reward computation
│ │ └── bias_reward.py
│ │
│ └── train/ # Training pipeline (GRPO)
│ │ └── grpo_train.py
│
│── .gitignore
```
