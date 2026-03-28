# GarryChess-DPO

### Preference-Optimized Policies for Realistic Grandmaster Chess

---

# Overview

**GarryChess-DPO** is a research project exploring whether **Direct Preference Optimization (DPO)** can be used to learn **fine-grained individual Grandmaster playing styles** from historical chess games while preserving stability, human-likeness, and tactical soundness.

Starting from **Maia-2**, a neural policy calibrated to human play around the ~2000 Elo level (when parameterized with the highest Elo band), we apply **preference-based learning** using pairwise move preferences extracted from Grandmaster games.

Rather than optimizing for engine strength, the objective is to:

> **Systematically prefer the moves a specific Grandmaster would choose over plausible alternatives while remaining close to a human-calibrated base policy.**

A central theme of this work is **human-centered chess AI**. The goal is not to surpass modern engines, but to understand and preserve *how elite humans play*.

We also study how preference-optimized policies interact with **shallow engine search at inference time**, revealing a clear tradeoff between **tactical strength and stylistic fidelity**.

---

# Research Questions

This project investigates several questions:

* Can **Direct Preference Optimization** capture the distinctive styles of individual Grandmasters from game data?
* How does DPO compare to **negative log-likelihood (NLL)** and **pairwise NLL** for stylistic alignment?
* Can stylistic fidelity be improved using **DPO with preference reweighting inspired by prospect-theoretic optimization (KTO)**?
* Do learned stylistic preferences persist when combined with **engine-based tactical filtering**?
* How does increasing search depth affect the balance between **playing strength and human-likeness**?
* At what point does **human-style play collapse under strong tactical pressure**?
* Can stylistic fidelity be **learned rather than heuristically specified**?
* Can preference learning move beyond pairwise comparisons and incorporate **partial preference orderings**?

---

# Core Contributions

## Preference-Based Grandmaster Style Learning

We reframe Grandmaster imitation as a **preference learning problem**.

For each position:

* the Grandmaster’s move is treated as the **preferred action**
* alternative legal moves form **rejected actions**

The policy is optimized via **Direct Preference Optimization (DPO)** to increase the likelihood of preferred moves while constraining divergence from the base **Maia-2** policy.

To our knowledge, this represents one of the **first applications of DPO to a structured board-game domain with discrete legal action constraints**.

---

## DPO vs NLL: Controlled Ablations

We compare three training strategies:

* **Standard supervised fine-tuning (NLL)** — next-move prediction
* **Pairwise supervised fine-tuning**
* **Direct Preference Optimization (DPO)**

All models share:

* identical datasets
* identical initialization
* identical evaluation protocols

### Key Findings

* DPO yields approximately **2× improvement in mean log-probability gap** between the Grandmaster’s chosen move and Maia’s next-best alternative.
* DPO achieves these gains with **minimal additional KL divergence** from the base Maia-2 policy.
* Preference optimization produces **stronger shallow-search performance with less stylistic drift**.

---

## Quantitative Human-Likeness Evaluation

Models are evaluated on **held-out Grandmaster positions (~20% of data)** stratified by game phase.

Metrics include:

* **Top-1 accuracy** on the Grandmaster move
* **Mean log-probability gap** (chosen vs rejected)
* **KL divergence** from the base Maia-2 policy
* **Phase-specific behavior statistics**

  * opening
  * middlegame
  * endgame
* **Opening fingerprint similarity** (ECO families)
* **Tactical volatility and sacrifice propensity proxies**

These metrics collectively measure **stylistic fidelity**, not just prediction accuracy.

---

## Inference-Time Tactical Filtering

Because Maia-2 operates below Grandmaster strength, we investigate whether tactical quality can be improved **without erasing learned style**.

At inference time:

1. **Stockfish** generates candidate moves via MultiPV.
2. Clear blunders are removed using a **centipawn-gap threshold** relative to the best engine move.
3. The learned policy **re-ranks the remaining candidates** according to stylistic preference.

Importantly:

> **Stockfish is never used during training.**

It only acts as a **tactical constraint during evaluation**.

---

## Strength vs Human-Likeness Tradeoff

We perform a systematic **search-depth sweep**:

```
Search depth ∈ {1, 2, 4, 8, 16}
```

For each depth we measure:

* engine-based Elo estimates
* all human-likeness metrics

This reveals a **Pareto frontier**:

* deeper search → stronger tactical play
* deeper search → weaker stylistic fidelity

DPO-trained policies retain stylistic alignment **longer under increasing search depth** than NLL baselines.

---

# Experiments

## 1. Grandmaster-Likeness Evaluation [Complete] (part of paper submitted to CoG)

Held-out positions evaluate how closely policies match the original Grandmaster decisions.

Metrics:

* Top-1 accuracy
* Log-probability gap statistics
* KL divergence vs base policy
* Opening distribution similarity
* Tactical volatility metrics

---

## 2. Strength Estimation 

Playing strength is estimated using a **local engine-vs-engine evaluation harness**.

Setup includes:

* fixed time controls
* color swapping
* Elo estimation with confidence intervals

Optional evaluation is performed using the **Lichess Bot API**.

---

## 3. Depth Ablation

By increasing tactical filtering strength at inference time, we isolate whether stylistic preferences learned via DPO persist under increasing tactical pressure.

This experiment reveals where **human style survives** and where it breaks down.

---

# Project Goals

Rather than pushing chess AI beyond human limits, **GarryChess-DPO** focuses on:

* faithfully modeling elite human decision-making
* preserving stylistic diversity
* enabling interpretability and historical analysis
* supporting pedagogical chess AI

All fine-tuned models and evaluation code are released publicly to enable further research.

---

# Repository Structure

```
garrychess-dpo/
├── src/                           # Training, inference, evaluation code
├── final_experiments_for_paper/   # Processed datasets, experiment outputs, graphs
├── maia2_models/                  # Base Maia-2 checkpoints (if required)
├── LICENSE                        # Apache 2.0 (code only)
├── NOTICE                         # Model artifact licensing notes
├── CITATION.cff                   # Citation metadata
└── README.md
```

---

# Reproducibility

The repository includes:

* PGN scraping and preprocessing pipelines
* NLL and DPO training scripts
* evaluation harnesses for style metrics
* Elo estimation scripts
* inference-time search wrappers

Exact experiment commands are documented in:

```
final_experiments_for_paper/README.md
```

Some experimental components are still under active development.

---

# Example Game

[![Magnus style win vs Lichess AI Level 8](https://lichess.org/ytgQOLw9.svg)](https://lichess.org/ytgQOLw9)

<details>
<summary><strong>Example PGN (click to expand)</strong></summary>

```pgn
[Event "casual blitz game"]
[Site "https://lichess.org/ytgQOLw9"]
[Date "2026.02.23"]
[White "lichess AI level 8"]
[Black "magnuscarlsenstyles"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. e5 d5
7. Bb5 Ne4 8. cxd4 Bb6 9. O-O O-O 10. Nc3 Bg4 11. Be3 f5
...
1/2-1/2
```

</details>

---

# GarryChess Website

You can play the models online:

**[https://www.garrychess.ai](https://www.garrychess.ai)**

---

# Lichess Bots

Example bots trained with Maia-2 + DPO:

* Magnus Carlsen
  [https://lichess.org/@/magnuscarlsenstyles](https://lichess.org/@/magnuscarlsenstyles)

* Alireza Firouzja
  [https://lichess.org/@/firouzjastyles](https://lichess.org/@/firouzjastyles)

* Praggnanandhaa
  [https://lichess.org/@/praggstyle](https://lichess.org/@/praggstyle)

Ratings fluctuate because bots are periodically updated and often play other bots.

---

# Licensing

Code
Apache License 2.0

Model weights
Released separately under different licensing terms.

See `LICENSE` and `NOTICE`.

---

# References

Key references used in this project:

**Human chess modeling**

* Maia: Human-Like Neural Network Chess Engines
  [https://arxiv.org/abs/2006.01855](https://arxiv.org/abs/2006.01855)

* Maia-2: A Unified Model for Human-AI Alignment in Chess
  [https://arxiv.org/abs/2409.20553](https://arxiv.org/abs/2409.20553)

* Learning Personalized Models of Human Behavior in Chess
  [https://arxiv.org/abs/2008.10086](https://arxiv.org/abs/2008.10086)

* Modeling Strong and Human-Like Gameplay with KL-Regularized Search
  [https://arxiv.org/abs/2112.07544](https://arxiv.org/abs/2112.07544)

---

**Preference learning**

* Direct Preference Optimization: Your Language Model Is Secretly a Reward Model
  [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)

* Learning from Human Preferences
  [https://arxiv.org/abs/1706.03741](https://arxiv.org/abs/1706.03741)

* Model Alignment as Prospect Theoretic Optimization (KTO)
  [https://arxiv.org/abs/2402.01306](https://arxiv.org/abs/2402.01306)

---

**Chess AI**

* Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)
  [https://arxiv.org/abs/1712.01815](https://arxiv.org/abs/1712.01815)

* DeepChess: End-to-End Deep Neural Network for Chess
  [https://arxiv.org/abs/1711.09667](https://arxiv.org/abs/1711.09667)

* KnightCap: TDLeaf(λ) learning for chess evaluation

* Leela Chess Zero
  [https://lczero.org](https://lczero.org)

* Stockfish Engine Documentation
  [https://official-stockfish.github.io/docs/](https://official-stockfish.github.io/docs/)

---

# Status

Active research project.

Initial paper submitted to **IEEE Conference on Games (CoG)**.

Further revisions and follow-up experiments are ongoing.

---
