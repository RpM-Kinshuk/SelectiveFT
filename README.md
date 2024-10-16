# LlaMAft
Codebase for LlaMA model fine tuning.

This project aims to demonstrate the effectiveness of Heavy-Tailed-Self-Regularization theory in accounting for the heavy tailed nature of weight matrices of deep neural networks to produce superior results in both training and fine-tuning pipelines. Our approach involves exploring the intersection of statistical mechanics and deep learning to integrate a novel method of fine-tuning deep neural networks, with Large Language Models (LLMs) being the primary focus.

With our method of sampling layers, the peak GPU memory usage and training times can be reduced to half as those in traditional full fine-tuning. Furthermore, our method can be combined with other Parameter Efficient Fine Tuning (PEFT) methods such as LoRA, QLoRA, DoRA etc. to achieve even superior results.

The approach works on both LLMs as well as theoretically, any deep neural network.
