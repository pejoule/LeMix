# LeMix
Concurrent Training and Serving of Large Language Models in Distributed Systems

## An illustration of Separate and Co-location strategies under concurrent workloads

- Separate training and inference on a 2-node cluster (each holds 3 sharded stages)
<p align="center">
  <img src="figure/separate_opportunity.png" width="100%" height="100%">
</p>

- Co-locate training and inference on a 2-node cluster (each holds 3 sharded stages)
<p align="center">
  <img src="figure/mix_opportunity.png" width="100%" height="100%">
</p>

- LeMix can dynamically adapt resource allocation (e.g., consolidate active nodes) and task priorities based on workload charactersitics and system conditions
<p align="center">
  <img src="figure/lemix_consolidation.png" width="60%" height="60%">
</p>


## Quickstart

### Setup Environment
- python 3.10.8
- pytorch 1.13.0+
- Install dependencies
```
pip install -r requirements.txt
```

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

### Autoregressive decoding (Llama2 & GPT): inference (prefilling & decoding) + training (S-PP)
- Separate with autoregressive decoding (hybrid batching w/ iteration=4) + GPipe S-PP scheduling (M=2)
<p align="center">
  <img src="figure/separate_opportunity_generation.png" width="60%" height="60%">
</p>

- Co-locate with autoregressive decoding (hybrid batching w/ iteration=4) + GPipe S-PP scheduling (M=2). $\emph{A}$ and $\emph{B}$ denote different requests. Subscript $\emph{d}$ represents a decode iteration and $\emph{p}$ represents a prefill operation.
<p align="center">
  <img src="figure/mix_opportunity_generation.png" width="60%" height="60%">
</p>

```
bash scripts/llama_generate.sh
bash scripts/dialogpt_generate.sh
```

