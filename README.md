# Chronic-Mem: Three-Layer Clinical Memory Metabolism Engine (Section 3)

This repository contains the standalone open-source implementation of the **Chronic-Mem Three-Layer Memory Metabolism Engine** (Section 3 of the paper), implementing **Clinical Significance-Aware Forgetting (CSAF)** and **Clinical Significance-Aware Progressive Compression (CSPC)**.

Only the core algorithmic framework ($F_1 \to F_2 \to F_3$) is included; benchmark harnesses, baseline runners, and downstream Agent wrappers are intentionally excluded.

## Authors

- **Ruiyu Yan**\*, **Renke Dai**\*, and **Yilin Kang**†  
  School of Computer Science (School of Artificial Intelligence), South-Central Minzu University  
  `202321241057@mail.scuec.edu.cn`, `2024110293@mail.scuec.edu.cn`, `ylkang@mail.scuec.edu.cn`  
  \*Equal contribution. †Corresponding author.

---

## Architecture Overview (Section 3)

```text
Raw Clinical Dialogue
   │
   ▼  Layer F1: Semantic Triage & Lifecycle Initialization (`chronic_mem/f1/`)
Memory Atoms m_i  ──►  UMLS TUI Mapping + Non-Homogeneous Decay Rate λ_i
   │
   ▼  Layer F2: Event Aggregation & Associated Latching (`chronic_mem/f2/`)
Clinical Events e_j ──►  Significance Gravity T(m_i, e_j) > ρ_event & Max-Pool λ_Event
   │
   ▼  Layer F3: Episode Resonance & Tri-State Metabolism (`chronic_mem/f3/`)
Episodes E_p       ──►  Resonance R(e_j, E_p) + Wake-up (Δt → 0) + EMA Prototype
                        Profile Metabolism f(E_p, t) ──► Active / Compressed (CSPC) / Forgotten
```

### 1. Layer $F_1$ — Semantic Triage & Lifecycle Initialization (`chronic_mem/f1/`)
- **Memory Atom Extraction (`f1/atoms.py`)**: Prompt template (`ATOM_EXTRACTION_SYSTEM_PROMPT`), pluggable LLM parser (`extract_atoms_from_dialogue`), offline heuristic clause splitter (`extract_atoms_heuristic`), and structured visit splitter (`atomize_event`).
- **UMLS Semantic Type Mapping (`f1/umls_mapper.py`)**: Maps atomic entities into clinical UMLS TUIs (`T037` allergy/injury, `T047` disease/syndrome, `T061` therapeutic procedure, `T121` pharmacologic substance, `T184` sign/symptom, etc.) and 16-d embeddings.
- **Non-Homogeneous Decay Rate $\lambda_i$ Rule Table (`f1/lambda_rules.py`)**:
  - **Critical (`99999.0d` / operational $\infty$)**: Core safety anchors (allergies `T037`, major surgeries, permanent flags)
  - **Decision (`120.0d`)**: Key clinical decisions (medications `T121`, procedures `T061`, disease recurrence)
  - **Phenotype (`30.0d`)**: Chronic disease baseline (`T047`), lifestyle/CGM monitoring logs, visit summaries
  - **Acute (`14.0d`)**: Transient symptoms (`T184`)
  - **Social (`3.0d`)**: Low-entropy greetings and conversational chatter
- **Urgency Scalar & Optional CPP (`f1/urgency.py`, `f1/cpp_model.py`, `f1/context.py`)**: Urgency scalar $\phi \in [0.8, 1.5]$ and optional MLP `ClinicalPersistencePredictor`.

### 2. Layer $F_2$ — Event Aggregation & Associated Latching (`chronic_mem/f2/`)
- **Significance Gravity (`f2/significance.py`)**:
  \[
  T(\text{Atom}, \text{Event}_j) = \frac{\gamma_{\text{med}} \cdot \mathrm{Sim}_{\text{med}} + \gamma_{\text{ctx}} \cdot \mathrm{Sim}_{\text{ctx}}}{\alpha + 1 / \lambda_{\text{Atom}}}
  \]
  Atoms with $T > \rho_{\text{event}}$ (`0.85`) attach to $\text{Event}_j$ with EMA prototype updates (`0.9 / 0.1`); otherwise a new event is opened.
- **Associated Latching (`f2/latching.py`)**:
  \[
  \lambda_{\text{Event}} = \max(\lambda_{m_1}, \dots, \lambda_{m_n})
  \]
  A single critical safety anchor shields the entire clinical visit context from premature decay.

### 3. Layer $F_3$ — Episode Resonance & Tri-State Metabolism (`chronic_mem/f3/`)
- **Cross-Time Resonance, Spatio-Temporal Wake-up & EMA Prototype (`f3/resonance.py`)**:
  \[
  R(e_j, E_p) = \gamma_{\text{med}} \cdot \mathrm{Sim}(v_{\text{med}}, w_{\text{med}}) + \gamma_{\text{ctx}} \cdot \mathrm{Sim}(v_{\text{ctx}}, w_{\text{ctx}})
  \]
  When $R(e_j, E_p) > \rho_{\text{episode}}$ (`0.65`), episode $E_p$ triggers **Spatio-Temporal Wake-up** (resetting `reference_time` to $t$, so $\Delta t \to 0$) and updates $(w_{\text{med}}, w_{\text{ctx}})$ via L2-normalized EMA.
- **Profile-Guided Real-Time Metabolism Score (`f3/metabolism.py`)**:
  \[
  f(E_p, t) = \mathrm{Res}(E_p, v_{\text{profile}}) \cdot \exp\!\left(-\frac{\Delta t}{\lambda_{\text{Episode}}}\right)
  \]
- **Tri-State CSPC Transitions & Physical Cleanup (`f3/cspc.py`)**:
  - **Active ($f > 0.7$)**: Retains full atomic memory and embeddings.
  - **Compressed ($0.3 < f \le 0.7$)**: Invokes **CSPC** to distill `PermanentLabel`s (`min_frequency=5`, with `T037` allergy exception at `freq >= 1`) and **physically deletes** redundant atoms + vectors where $f(m_i, t) < \tau_{\text{prune}}$ (`0.3`).
  - **Forgotten ($f \le 0.3$)**: Physically deletes all remaining `AtomNode`s and `event:*` / `episode_agg:*` vectors, retaining **only** the `PermanentLabel` index.
- **Adaptive Retrieval Granularity Controller (`f3/controller.py`)**: Dynamically switches query-time context granularity across `atoms` (`active`), `summaries` (`compressed`), and `labels` (`forgotten`).

---

## Repository Layout

```text
chronic-mem/
├── chronic_mem/
│   ├── __init__.py
│   ├── types.py             # AtomNode, EventNode, EpisodeNode, PermanentLabel, EmbeddingStore
│   ├── geometry.py          # L2 normalization & cosine-to-[0, 1] mapping
│   ├── encoding.py          # Lazy BERT CLS encoder (`bert-base-uncased` + offline fallback)
│   ├── engine.py            # End-to-end ChronicMemEngine (CSAFHierarchicalMetabolizer)
│   ├── f1/                  # F1: Semantic Triage & Lifecycle Initialization
│   │   ├── atoms.py         # Dialogue-to-Atoms extraction Prompt + LLM/heuristic splitters
│   │   ├── umls_mapper.py   # UMLS TUI mapping & 16-d type embeddings
│   │   ├── lambda_rules.py  # 5-tier non-homogeneous λ_i rule configuration table
│   │   ├── urgency.py       # Clinical urgency scalar φ detector
│   │   ├── cpp_model.py     # Optional neural λ_i predictor (MLP)
│   │   └── context.py       # F1 runtime triage context
│   ├── f2/                  # F2: Event Aggregation & Associated Latching
│   │   ├── significance.py  # Significance Gravity T(Atom, Event_j) & ρ_event clustering
│   │   └── latching.py      # Max-Pooling λ_Event = max(λ_m1, ..., λ_mn)
│   └── f3/                  # F3: Episode Resonance & Tri-State Metabolism
│       ├── resonance.py     # R(e_j, E_p), Spatio-Temporal Wake-up (Δt -> 0), EMA prototype
│       ├── metabolism.py    # v_profile metabolism score f(E_p, t) & Tri-State thresholds
│       ├── cspc.py          # CSPC PermanentLabel distillation & physical vector deletion
│       └── controller.py    # Adaptive Retrieval Granularity Controller
├── examples/
│   ├── events_small.json    # Synthetic two-patient clinical trajectory fixture
│   └── demo_section3.py     # End-to-end F1 -> F2 -> F3 walkthrough
├── tests/
│   └── test_csaf_hierarchical.py
└── docs/
    └── IMPLEMENTATION.md    # Hyperparameter & rule tables for reproducibility
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run core unit & integration tests (offline deterministic encoder)
CHRONIC_MEM_ENCODER=hash python tests/test_csaf_hierarchical.py

# Run end-to-end Section 3 walkthrough (F1 -> F2 -> F3)
CHRONIC_MEM_ENCODER=hash python examples/demo_section3.py
```

To encode with `bert-base-uncased` instead of the offline hash encoder, omit `CHRONIC_MEM_ENCODER=hash` (or place local Hugging Face weights under `./bert/`).

---

## License

MIT License. Copyright (c) 2026 Ruiyu Yan, Renke Dai, and Yilin Kang. See `LICENSE`.  
Research prototype only; not a medical device.
