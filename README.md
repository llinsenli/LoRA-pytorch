# LoRA: Low-Rank Adaptation of Large Language Models

This repository implements the Low-Rank Adaptation (LoRA) approach as described in the seminal paper by Edward J. Hu et al. This method is designed to efficiently fine-tune large language models by adapting only a small subset of their parameters.

## Original Paper
Please reference the original work if you use this implementation:
```
@article{hu2021lora,
  title={Lora: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

## Background

### Fine-tuning Large Language Models
Fine-tuning is a critical process where pre-trained models are slightly adjusted to specialize them for specific tasks. This adjustment can be applied to all or part of the model's parameters, depending on the task requirements and the available data.

#### Advantages of Fine-tuning
- **Efficiency**: Leverages pre-learned features, reducing the need for extensive training data.
- **Effectiveness**: Helps achieve high performance on specific tasks by refining general capabilities into specialized skills.

### Challenges with Fine-tuning
Fine-tuning entire models can be computationally expensive and prone to overfitting, especially when only a small dataset is available for the new task.

## LoRA: Concept and Benefits

### How LoRA Works
LoRA introduces a novel approach to fine-tuning by decomposing the original weight matrix $W$ into two smaller matrices $A$ and $B$. Here's the mathematical model:
- Assume $W$ is a weight matrix in the model of dimensions $d \times h$.
- Instead of updating $W$, LoRA freezes $W$ and introduces matrices $A$ of size $d \times r$ and $B$ of size $r \times h$, where $r$ is the rank and $r \ll d, h$.
- The effective update is $W' = W + AB$, where $AB$ represents the low-rank modification to $W$.

### Benefits of Using LoRA
- **Parameter Efficiency**: As $r$ is much smaller than $d$ and $h$, $A$ and $B$ have significantly fewer parameters than $W$, reducing the number of trainable parameters.
- **Flexibility**: Allows for rapid adaptation to new tasks without the computational burden of retraining the entire model.
- **Preservation of Pre-trained Dynamics**: By keeping $W$ frozen, the original pre-trained dynamics are preserved, minimizing catastrophic forgetting.

### Principle Behind LoRA
The foundational insight of LoRA is that many real-world data and the corresponding weights in deep models exhibit low-rank structures. This means that the effective information can be captured with fewer parameters than in the full matrix $W$:
- **Mathematical Justification**: If $W$ can be approximated using Singular Value Decomposition (SVD) with a small number of singular values, then $AB$ (with $A$ and $B$ representing the truncated SVD) can capture the essential transformations in $W$ effectively.

## Implementation Example

A practical implementation of LoRA is provided in the `LoRA_DistilBERT_IMDB.ipynb` notebook, where LoRA is applied to the DistilBERT model for text classification on the IMDb dataset. 


