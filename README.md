\# Deep Reinforcement Learning for RIS Phase-Shift Optimization in 6G Networks



This repository contains the code, trained models, and evaluation pipeline for the comparative analysis of Deep Reinforcement Learning (DDPG, SAC) and Deep Unfolding for Reconfigurable Intelligent Surface (RIS) phase-shift optimization in 6G networks.



\##  Abstract

This project evaluates model-free Deep Reinforcement Learning against model-driven Deep Unfolding for RIS phase-shift optimization under imperfect Channel State Information (CSI) and discrete hardware constraints. We benchmark DDPG, SAC (with a Gumbel-Softmax 3-bit codebook), and a Deep Unfolding network based on Riemannian Conjugate Gradient.



\## 📁 Project Structure

```text

Deep-Reinforcement-Learning-for-RIS-Phase-Shift-Optimization/

│

├── ris\_env.py                  # The unified 6G RIS simulation environment (State: 1056, Action: 64)

├── ao\_baseline.py              # Alternating Optimization (Riemannian Gradient Ascent) baseline

├── generate\_dataset.py         # Script to generate the 10,000 sample AO ground-truth dataset

├── evaluate.py                 # Unified evaluation pipeline for M1, M2, and M3

├── plot\_final\_figure2.py       # Matplotlib script to generate the robustness graph

├── generate\_figure1.py         # Matplotlib script to generate the architecture diagram

│

├── models/                     # Trained model weights (.pth)

│   ├── m1\_ddpg\_actor.pth

│   ├── m2\_sac\_actor.pth

│   └── m3\_unfolding.pth

│

├── data/                       # Generated datasets

│   └── ao\_ground\_truth.pt

│

├── results/                    # Evaluation outputs

│   ├── evaluation\_results.txt

│   └── raw\_eval\_data.csv

│

└── figures/                    # Generated plots for the manuscript

&#x20;   ├── figure1\_architecture.png

&#x20;   └── figure2\_robustness.png

