# Chronic-Mem (Section 3) — Implementation Details for Reproducibility

> **Scope:** Default hyperparameters, UMLS TUI rules, non-homogeneous $\lambda_i$ configuration, $F_2$ Significance Gravity, and $F_3$ Tri-State CSPC metabolism constants in the open-source `chronic_mem` core engine.

---

## A. Global Survival Function & Text Encoding (`chronic_mem/encoding.py`, `chronic_mem/f3/metabolism.py`)

| Item | Specification | Module |
|------|---------------|--------|
| Atom query survival | \(f(m_i, t) = \mathrm{Res}(m_i, q) \cdot \exp(-\Delta t_i / \lambda_i)\) | `chronic_mem/engine.py` |
| Episode metabolism score | \(f(E_p, t) = \mathrm{Res}(E_p, v_{\mathrm{profile}}) \cdot \exp(-\Delta t / \lambda_{\mathrm{Episode}})\); \(\Delta t = t - \texttt{reference\_time}\) | `chronic_mem/f3/metabolism.py` |
| Resonance mapping | raw cosine \(\in [-1, 1] \mapsto (\cos + 1)/2 \in [0, 1]\) | `chronic_mem/geometry.py` |
| Text encoder | `bert-base-uncased` CLS (768-d); local `./bert/` if present; deterministic 768-d fallback when offline | `chronic_mem/encoding.py` |
| Time unit & floor | Days (`.total_seconds() / 86400.0`), numerical floor `lam + 1e-8` | throughout |

---

## B. Layer $F_1$: Semantic Triage & Lifecycle Initialization (`chronic_mem/f1/`)

### B.1 Memory Atom Extraction (`f1/atoms.py`)
- **Atom kinds:** `diagnosis`, `symptom`, `treatment`, `medication`, `lifestyle`, `people`, `follow_up`, `summary`, `social`.
- **Extraction modes:**
  1. LLM prompt extraction (`ATOM_EXTRACTION_SYSTEM_PROMPT` + `extract_atoms_from_dialogue(..., llm_fn=...)`).
  2. Offline heuristic clause extraction (`extract_atoms_heuristic`).
  3. Structured clinical visit decomposition (`atomize_event`).

### B.2 Embedded UMLS TUI Vocabulary (`f1/umls_mapper.py`)

| TUI | Semantic Type | Primary Rule Trigger |
|-----|---------------|----------------------|
| `T037` | Injury or Poisoning (**Allergy safety anchor**) | `allergy` / `allergic` / `hypersensitivity` / `过敏` |
| `T061` | Therapeutic or Preventive Procedure | `kind=treatment` or surgery/procedure/therapy keywords |
| `T047` | Disease or Syndrome | `kind=diagnosis` or diagnosis/disease/syndrome keywords |
| `T184` | Sign or Symptom | `kind=symptom` or symptom/pain/fever/cough keywords |
| `T121` | Pharmacologic Substance | `kind=medication` or drug/pill/prescription keywords |
| `T000` | Default / Unclassified | fallback when no clinical type matches |

*(Full 21-type clinical TUI table + `T000` is embedded in `UMLS_TUI_MAPPING` with `nn.Embedding(22, 16)`.)*

### B.3 Non-Homogeneous Decay Rate $\lambda_i$ Configuration (`f1/lambda_rules.py`)

| Tier | $\lambda_i$ (days) | Clinical Decision Value |
|------|--------------------|-------------------------|
| **Critical** | **99999.0** ($\infty$) | Core safety anchors: allergies (`T037` / markers), major surgeries, `is_permanent` |
| **Decision** | **120.0** | Medications (`T121`), therapeutic procedures (`T061`), disease recurrence |
| **Phenotype** | **30.0** | Chronic diagnoses (`T047`), lifestyle/CGM logs, follow-up & visit summaries |
| **Acute** | **14.0** | Transient acute symptoms (`T184`) |
| **Social** | **3.0** | Greetings, polite acknowledgments, low-entropy small talk |

---

## C. Layer $F_2$: Event Aggregation & Associated Latching (`chronic_mem/f2/`)

### C.1 Significance Gravity (`f2/significance.py`)

\[
T(\mathrm{Atom}, \mathrm{Event}_j) = \frac{\gamma_{\mathrm{med}} \cdot \mathrm{Sim}_{\mathrm{med}} + \gamma_{\mathrm{ctx}} \cdot \mathrm{Sim}_{\mathrm{ctx}}}{\alpha + 1 / \lambda_{\mathrm{Atom}}}
\]

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\gamma_{\mathrm{med}}$ | **0.8** | Weight on UMLS TUI embedding similarity |
| $\gamma_{\mathrm{ctx}}$ | **0.2** | Weight on contextual text embedding similarity |
| $\alpha$ | **0.5** | Gravity denominator offset |
| $\rho_{\mathrm{event}}$ | **0.85** | Strict threshold `T > 0.85` to join existing `Event_j` |
| Prototype EMA | **0.9 / 0.1** | L2-normalized EMA (skipped for generic `summary`/`follow_up`/`people`/`T000`) |

### C.2 Associated Latching (`f2/latching.py`)

\[
\lambda_{\mathrm{Event}} = \max(\lambda_{m_1}, \dots, \lambda_{m_n})
\]

---

## D. Layer $F_3$: Episode Resonance & Tri-State Metabolism (`chronic_mem/f3/`)

### D.1 Cross-Time Episode Resonance & Wake-up (`f3/resonance.py`)

\[
R(e_j, E_p) = 0.8 \cdot \mathrm{Sim}(v_{\mathrm{med}}, w_{\mathrm{med}}) + 0.2 \cdot \mathrm{Sim}(v_{\mathrm{ctx}}, w_{\mathrm{ctx}})
\]

| Mechanism | Default | Behavior |
|-----------|---------|----------|
| Merge threshold $\rho_{\mathrm{episode}}$ | **0.65** | `R > 0.65` aligns event $e_j$ into episode $E_p$ regardless of elapsed time |
| Spatio-Temporal Wake-up | `reference_time <- t` | Resets $\Delta t \to 0$ (and restores `compressed` $\to$ `active`) |
| Prototype EMA | **0.9 / 0.1** | $w_p \leftarrow \mathrm{normalize}(0.9\,w_p + 0.1\,v_{e_j})$ |

### D.2 Tri-State Thresholds, CSPC & Adaptive Retrieval (`f3/metabolism.py`, `f3/cspc.py`, `f3/controller.py`)

| State | Condition on $f(E_p, t)$ | Storage Action (CSPC) | Adaptive Retrieval Granularity |
|-------|--------------------------|-----------------------|--------------------------------|
| **Active** | $f > 0.7$ | Retain all `AtomNode`s and vectors | `atoms`: Event headers + ranked atomic lines ($f \ge 0.1$) |
| **Compressed** | $0.3 < f \le 0.7$ | Distill `PermanentLabel`s (`min_frequency=5`, `T037` allergy at `freq >= 1`); physically delete atoms + vectors with $f(m_i, t) < 0.3$ | `summaries`: `PermanentLabel`s + Top-$k$ Event summaries |
| **Forgotten** | $f \le 0.3$ | Distill `PermanentLabel`s; physically delete all `AtomNode`s + `event:*` + `episode_agg:*` vectors | `labels`: `PermanentLabel` index ONLY |
