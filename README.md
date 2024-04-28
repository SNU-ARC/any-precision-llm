# Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs [[Paper](http://www.arxiv.org/pdf/2402.10517)]


## Overview

Any-precion LLM is a memory-efficient and cost-effective solution for deployment of multiple, different sized LLMs. Specifically, any-precision LLM redues the memory cost of deplying multiple, different-sized LLMs by overlaying LLMs quantized to varying bit-widths, such as 3, 4, ..., n bits, into a memory footprint comparable to a single n-bit LLM. This includes a lightweight any-precision quantization technique for LLMs called incremental upscaling, and a specialized software engine for efficient serving, which is equipped with a custom CUDA kernel supporting bitplane-based weight representation.
<div align=center>	
<p align="center">
<img width="500" src="./figures/incremental_upscaling.png">
  
Illustration of incremental upscaling scheme
</p>

<p align="center">
<img width="500" src="./figures/software_engine.png">
</p>

Illustration of specialized software engine for any-precision LLM
</div>

## How to use any-precision LLMs

