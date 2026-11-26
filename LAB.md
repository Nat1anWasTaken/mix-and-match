<!--Copyright © 2026 Mu-Tien Kuo-->

## Description

Data is one of the most important parts of training a model. This lab aims to serve as an introductory course to data curation and mixing. You are find a data recipe which reaches ≤ 1.00 bits-per-bytes on Paloma’s 9 suites’ macro average using an 1B model with Llama 3.2 1B’s architecture. 

You will compete on reducing the amount of tokens required to achieve the BPB goal. To accomplish this, you may freely use any source of data, such as published pretraining datasets (e.g., Dolma, Red Pajama, FineWeb, FinePDFs, etc.), common crawl checkpoints, or even crawl data by yourself. Nonetheless, you should run Paloma decontamination on your training data following Paloma Appendix C.1’s instructions. 

We define token count as the amount of tokens that was used to directly update the weights of your model. This means that if you train on the same data for 2 epochs, you “train on” 2x the token dataset count. Yet, as an interesting twist, you are allowed to repeat up to 10% of your training data for free. For example, say you train on 200B tokens, you can choose 20B token  to repeat without them counting towards your training total. 

All random generators should have their seeds fixed to 42. Seed-sweeping is not an intended result of this lab. 

## Compute Grants

To encourage experiment planning, each *pair* will receive compute grants in a ladder-like fashion. Each grant permits one run using **up to** its assigned H200-hour allowance.

The grant levels are based on the compute required by a well-optimized PyTorch trainer using `torch.compile`, FlashAttention-3, and sequence packing, training reference models at 200 tokens per parameter. The allowances are intended to give pairs some breathing room for differences in implementation.

H200-hours count compute across all GPUs used in a run. For example, using 8 H200s for 2 hours consumes 16 H200-hours.

| Grant allowance per run | Ref scale | Ref tokens | #grants | Availability |
| --- | --- | --- | --- | --- |
| 40 H200-hours | 150M | 30B | Inf | Throughout both weeks |
| 110 H200-hours | 300M | 60B | 10 | Throughout both weeks |
| 266 H200-hours | 500M | 100B | 5 | Week 2 |
| 2160 H200-hours | 1B | 200B | 3 | Week 2 |

The reference scales describe the intended size of each compute allowance. They do not restrict your experiments to those parameter counts. You may choose other model sizes using the architecture scaling equation.

You may use a grant for a smaller experiment or stop a run early. Nonetheless, each launched experiment consumes a grant at its assigned level. These grants are intended to provide a certain number of experimental opportunities, rather than a compute wallet that can be divided into unlimited replacement runs. Resuming an interrupted run from its checkpoint is considered part of the same run.

You are free to choose the model sizes and training horizons of your exploratory experiments within their grant allowances. The required 750M calibration run should use the scaling equation and train for 200 tokens per parameter.

## Evaluation

Use Flock Evals to measure your 1B model’s performance on Paloma. The lab target is **≤ 1.00 bits per byte (BPB)** on the macro average across Flock’s supported 9 Paloma sources. Within each source, take the unweighted mean of its domain BPB scores. Then average the source scores, giving each source equal weight.

Accept the Paloma dataset terms before your first run. Evaluate a pinned model revision in float32 on the full test split. Paloma dataset · Flock evaluation guide

```bash
flock eval run \
  --suite paloma \
  --model "your-org/your-model@your-revision" \
  --dtype float32 \
  --out results/paloma
```

The current Flock suite covers 309 domains across 9 sources. Use the **9-source macro average** to compare experiments and include the source count when reporting it. The **1.00 BPB target** requires scores for all 9 sources.

## Submissions

This lab does not require manual submissions. Instead, the OJ will read the following summary metrics in Flock’s `lab7-data-curation` project:

- `oj_total_tokens` : The OJ reports this as the final score. This number should be the post-repetition-discount figure. E.g., if you train on 200B tokens with 20B repeated, report 200B instead of 220B.
- `total_tokens` : This is for human reference, please report the total number of tokens the model has been trained on.
- `paloma_ppl` : The OJ checks this figure to ensure your run qualifies.
