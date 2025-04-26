# Multi-task Two-stage Learning Framework (MTLF)

## Introduction
We revisit the multi-task collaborative learning in MSA. Specifically, we incorporate a teacher-student learning paradigm (knowledge distillation) to address the challenge of scarce unimodal labels, and propose the TLS as an alternative to traditional self-supervised label generation techniques. This involves leveraging the multimodal representation (teacher) to guide the learning process of the unimodal representation (student).

## Usage
python3.8    
torch-1.12.0+cu116

python train.py

## Overall framework
![MTLF](/Image/overall_framework.jpg)
