# LeMix
Concurrent Training and Serving of Large Language Models in Distributed Systems

## Abstract

The "train-then-inference" paradigm is commonly adopted in the deployment of large language models and other deep learning models, resulting in GPU under-utilization (see the below profiling results on LMSYS workload traces) and inconsistent model update in distributed systems.  
<p align="center">
  <img src="figure/GPU_util.png" width="100%" height="100%">
</p>

Our motivation study reveals that these inefficiencies stem from dynamic request arrivals during serving and workload heterogeneity in pipeline-parallel training.
- Separate training and inference on a 2-node cluster (each holds 3 sharded stages)
<p align="center">
  <img src="figure/separate_opportunity.png" width="100%" height="100%">
</p>


We first propose a baseline strategy, NaiveMix, which assigns tasks to nodes using a fair Round-Robin (RR) policy based on their in-queue order.
- Co-locate training and inference on a 2-node cluster (each holds 3 sharded stages)
<p align="center">
  <img src="figure/mix_opportunity.png" width="100%" height="100%">
</p>

To further improve the coarse-grained scheduling limitations in NaiveMix, we propose LeMix that can dynamically adapt resource allocation based on workload characteristics and system conditions by understanding task-specific behaviors and resource contention
across shared nodes. LeMix effectively balances the trade-offs between utilization, serving quality, and serving responsiveness.
- LeMix can consolidate active nodes and prioritizes tasks based on workload charactersitics and system conditions
<p align="center">
  <img src="figure/lemix_consolidation.png" width="60%" height="60%">
</p>


## Installation

### Setup Environment
- python 3.10
- pytorch 1.13.0+
- Install dependencies
```
conda create -n lemix python=3.10
# Optional: Install CUDA via conda for a smoother installation experience and profiling experiments
pip install -r requirements.txt
```
For more details on installing CUDA via conda, refer to the [CUDA Installation Guide by NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation).

## Evaluation

### Main evaluation (Llama2 & GPT): inference (prefilling) + training (A-PP)
```
bash scripts/main_llama.sh
bash scripts/main_dialogpt.sh
```

### Heterogeneity study (GPT): inference (prefilling) + training (A-PP)
```
bash scripts/LH_study.sh
```

### Ablation study (Llama2): inference (prefilling) + training (A-PP)
```
bash scripts/ablation_study.sh
```

### Sensitivity study (Llama2): inference (prefilling) + training (A-PP)
```
bash scripts/parameter_study.sh
```

### Synchronous PP (Llama2 & GPT): inference (prefilling) + training (S-PP)
- Separate with GPipe S-PP scheduling (M=2)
<p align="center">
  <img src="figure/separate_opportunity_gpipe.png" width="100%" height="100%">
</p>

- Co-locate with GPipe S-PP scheduling (M=2)
<p align="center">
  <img src="figure/mix_opportunity_gpipe.png" width="100%" height="100%">
</p>

- Separate with 1F1B S-PP scheduling (M=2)
<p align="center">
  <img src="figure/separate_opportunity_1f1b.png" width="100%" height="100%">
</p>

- Co-locate with 1F1B S-PP scheduling (M=2)
<p align="center">
  <img src="figure/mix_opportunity_1f1b.png" width="100%" height="100%">
</p>

```
bash scripts/llama_SPP.sh
bash scripts/dialogpt_SPP.sh
```
- We compare LeMix with other methods in inference loss, throughput, and response time SLO attainment.
<p align="center">
  <img src="figure/loss_spp.png" width="70%" height="70%">
</p>
<p align="center">
  <img src="figure/throughput_spp.png" width="70%" height="70%">
</p>
<p align="center">
  <img src="figure/SLO_response_spp.png" width="70%" height="70%">
</p>

### Autoregressive decoding (Llama2 & GPT): inference (prefilling & decoding) + training (S-PP)
LeMix supports a hybrid iteration-level batching of prefilling and decoding workloads for serving. 
<p align="center">
  <img src="figure/continuous_batching.png" width="50%" height="50%">
</p>

- Separate with autoregressive decoding (hybrid batching w/ iteration=4) + GPipe S-PP scheduling (M=2)
<p align="center">
  <img src="figure/separate_opportunity_generation.png" width="60%" height="60%">
</p>

- Co-locate with autoregressive decoding (hybrid batching w/ iteration=4) + GPipe S-PP scheduling (M=2). *A* and *B* denote different requests. Subscript *d* represents a decode iteration and *p* represents a prefill operation.
<p align="center">
  <img src="figure/mix_opportunity_generation.png" width="60%" height="60%">
</p>

```
bash scripts/llama_generate.sh
bash scripts/dialogpt_generate.sh
```
- LEMIX consistently achieves the lowest latency in both prefilling (TTFT) and decoding (TBT) phases across all request rates. Specifically, task planning (§4.2) and resoure allocation (§4.3) primarily affect prefilling performance, while prioritization (§4.3) and runtime scheduling (§4.4) contribute significantly to the decoding phase compared to other co-location methods.
<!-- <p align="center">
  <img src="figure/latency(generate)-lambda_nodes=4.png" alt="Figure 1" width="49%">
  <img src="figure/latency(generate)-batch_nodes=4.png" alt="Figure 2" width="49%">
</p> -->
<p align="center">
  <img src="figure/latency(generate)-lambda_nodes=4.png" width="70%" height="70%">
</p>
<p align="center">
  <img src="figure/latency(generate)-batch_nodes=4.png" width="70%" height="70%">
</p>

### Data parallelism (GPT): inference (prefilling) + training (DP)
```
bash scripts/dialogpt_dp.sh
```
- We compare LeMix with other methods in inference loss, throughput, and response time SLO attainment.
<p align="center">
  <img src="figure/loss_dp.png" width="70%" height="70%">
</p>
<p align="center">
  <img src="figure/throughput_dp.png" width="70%" height="70%">
</p>
<p align="center">
  <img src="figure/SLO_response_dp.png" width="70%" height="70%">
</p>