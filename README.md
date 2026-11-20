# Chronic-Mem: Enhancing Medical Agents Memory via Hierarchical Clinical-Semantic Significance Metabolism for Chronic Disease Management

As chronic disease management moves toward decentralized home care, medical agents must preserve accurate longitudinal patient histories while filtering noisy interaction data. Existing memory systems are largely time-centric and apply homogeneous decay, so they struggle to separate transient social details from persistent clinical anchors such as allergies, medications, and chronic conditions.

**Chronic-Mem** is a clinical significance-aware memory metabolism framework that shifts memory management from physical time to semantic criticality. It combines semantic triage ($F_1$), event aggregation ($F_2$), and a tri-state metabolic pipeline ($F_3$) to retain clinically essential evidence while compressing low-value context.

> This repository releases the **core $F_1\!\to\!F_2\!\to\!F_3$ engine** only. Benchmarks and agent wrappers are not included.

## Authors

**Ruiyu Yan**\*, **Renke Dai**\*, and **Yilin Kang**†  
School of Computer Science (School of Artificial Intelligence), South-Central Minzu University  
`202321241057@mail.scuec.edu.cn`, `2024110293@mail.scuec.edu.cn`, `ylkang@mail.scuec.edu.cn`  
\*Equal contribution. †Corresponding author.

## Pipeline

```text
Dialogue ──► F1 Semantic Triage      Atoms + UMLS TUI + non-homogeneous λ_i
         ──► F2 Event Aggregation    Significance Gravity + Max-Pool λ_Event
         ──► F3 Episode Metabolism   Resonance / Wake-up / Active·Compressed·Forgotten
```

| Layer | Role | Key modules |
|-------|------|-------------|
| **$F_1$** | Atomize dialogue; map UMLS types; assign decay $\lambda_i$ | `chronic_mem/f1/` |
| **$F_2$** | Cluster atoms into events via significance gravity | `chronic_mem/f2/` |
| **$F_3$** | Episode resonance, CSPC compression, adaptive retrieval | `chronic_mem/f3/` |

Hyperparameters and rule tables: [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md).

## Quick Start

```bash
pip install -r requirements.txt

CHRONIC_MEM_ENCODER=hash python tests/test_csaf_hierarchical.py
CHRONIC_MEM_ENCODER=hash python examples/demo_section3.py
```

Omit `CHRONIC_MEM_ENCODER=hash` to use `bert-base-uncased` (or place weights under `./bert/`).

## License

MIT © 2026 Ruiyu Yan, Renke Dai, and Yilin Kang. Research prototype only; not a medical device.
