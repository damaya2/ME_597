# Representing Adaptive Spacecraft Guidance Policies with Large Language Models in the Earth–Moon CRTBP

**Author:** Daniel Amaya 
**Course:** ME 597  

---

## Overview

This repository investigates the use of fine-tuned Large Language Models (LLMs) as closed-loop spacecraft guidance controllers within the **Earth–Moon Circular Restricted Three-Body Problem (CRTBP)**. The framework integrates non-linear orbital dynamics, classical proportional-derivative (PD) and proportional-integral-derivative (PID) feedback control baselines, automated cislunar dataset generation, QLoRA instruction tuning on Llama-3.1-8B-Instruct, and real-time closed-loop LLM trajectory execution.

The primary objective is to evaluate whether an LLM can learn the map between spatial tracking errors and corrective three-axis control actions while maintaining trajectory tracking stability and disturbance recovery across cislunar mission profiles.

---

## Directory Architecture

.
├── Code/                      # Numerical dynamics, controllers, & script runners
│   ├── CRTBP.py               # Earth-Moon CRTBP equations of motion & integrators
│   ├── Controller.py          # Classical baseline controllers (PD / PID)
│   ├── LLM_Controller.py      # Real-time closed-loop LLM controller interface
│   ├── LLM_Guidance.py        # LLM token formatting & inference routines
│   ├── Fine_Tuning.slurm      # NCSA Delta HPC training submission script
│   └── *.sh / *.slurm         # Automated workflow execution & simulation scripts
├── PD_LLM/                    # PD guidance baseline experiment iteration
├── PID_LLM/                   # Current PID guidance adaptive controller iteration
├── PID_OLD_LLM/               # Deprecated PID implementation (legacy reference)
├── Logs/                      # Training JSONL datasets, checkpoints, & Tensorboard events
├── Plots/                     # Output 2D/3D trajectory plots, GIFs, & loss metrics
├── ME_597_Final_Paper.pdf     # Final research project paper
└── ME 597 Final Presentation.pptx

> **Note on Model Variants:** `PD_LLM` and `PID_LLM` are independent, stand-alone experimental implementations evaluating different baseline controller guidance strategies. `PID_OLD_LLM` is retained strictly as an archival version of `PID_LLM`.

---

## Environment Setup & Prerequisites

### 1. Requirements

* **Python Version:** 3.9 (Recommended for full compatibility with NCSA Delta cluster dependencies)
* **Hugging Face Token:** An active Hugging Face User Access Token with granted access to download Meta's Llama-3.1-8B-Instruct model weights.

### 2. Required Libraries

Install the required packages using pip:

pip install torch transformers peft trl accelerate bitsandbytes datasets numpy scipy matplotlib pandas scikit-learn

---

## Execution Workflow

The end-to-end pipeline consists of six primary steps:

[ CRTBP Simulation Setup ] ──► [ Reference Trajectories ] ──► [ Dataset Generation ]
                                                                      │
[ Plotting & Metrics ] ◄── [ Closed-Loop LLM Guidance ] ◄── [ QLoRA Fine-Tuning ]

### Step 1 & 2: CRTBP Simulation & Reference Trajectories
Nonlinear Earth–Moon CRTBP state vectors and target reference orbits are defined using Code/CRTBP.py and Code/Controller.py.

### Step 3: Dataset Creation
Supervised instruction-tuning .jsonl datasets are created by pairing state tracking errors (e_x, e_y, e_z, e_vx, e_vy, e_vz) with nominal baseline control actions:

bash Code/generate_dataset.sh

Outputs are saved to the Logs/ directory.

### Step 4: LLM Fine-Tuning
Fine-tune Llama-3.1-8B-Instruct using QLoRA. Pass your Hugging Face authentication token when launching:

export HF_TOKEN="your_huggingface_token_here"
sbatch Code/Fine_Tuning.slurm

Checkpoints, adapter weights, and TensorBoard logs will populate inside Logs/.

### Step 5 & 6: Closed-Loop Guidance & Evaluation
Execute the closed-loop controller using the fine-tuned LLM adapter (PID_LLM or PD_LLM) to generate real-time control actions step-by-step:

python Code/LLM_Controller.py --adapter_path Logs/final_adapter

Simulation outputs, performance metrics, loss curves, and 2D/3D orbital plots will be generated and saved into the Plots/ directory.

---

## Running on NCSA Delta HPC

This repository includes custom .slurm and .sh scripts tailored for high-performance execution on the NCSA Delta GPU cluster.

For system allocations, module environments, and cluster-specific job setup guidelines, refer to the official NCSA Delta System Documentation (https://docs.ncsa.illinois.edu/systems/delta/en/latest/).

---

## Contact

**Author:** Daniel Amaya
**Advisor:** Dr. Hiroyasu Tsukamoto  
**Institution:** University of Illinois Urbana-Champaign  
**Department:** Aerospace Engineering / Mechanical Science and Engineering