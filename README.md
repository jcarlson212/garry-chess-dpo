# **GarryChess-DPO**

### *Preference-Optimized Policies for Realistic Grandmaster Chess*

---

## Overview

**GarryChess-DPO** is a research project exploring whether **Direct Preference Optimization (DPO)** can be used to learn **fine-grained, individual Grandmaster playing styles** from historical chess games—while preserving stability, human-likeness, and tactical soundness.

Starting from **Maia-2**, a neural policy calibrated to human play around the ~2000 Elo level, we apply **preference-based learning** using pairwise move preferences extracted from Grandmaster games. Rather than optimizing for engine strength, the objective is to **systematically prefer the moves a specific Grandmaster would choose over plausible alternatives**, while remaining close to the base human-calibrated policy.

A central theme of this work is **human-centered chess AI**: we are not attempting to surpass modern engines, but to understand and preserve *how* elite humans play. We study how preference-optimized policies interact with **shallow engine search at inference time**, revealing a clear tradeoff between tactical strength and stylistic fidelity.

---

## Research Questions

* Can **Direct Preference Optimization** capture the distinctive styles of individual Grandmasters from game data?
* How does DPO compare to **supervised fine-tuning (SFT)** and pairwise SFT for stylistic alignment?
* Do learned stylistic preferences persist when combined with **engine-based tactical filtering**?
* How does increasing search depth affect the balance between **playing strength** and **human-likeness**?
* Where does human-style play begin to collapse under strong tactical pressure?

---

## Core Contributions

### **Preference-Based Grandmaster Style Learning**

We reframe Grandmaster imitation as a **preference-learning problem**. For each position, the Grandmaster’s played move is treated as a preferred action relative to strong alternatives, and the policy is optimized via DPO to increase this relative likelihood while constraining divergence from the base Maia-2 policy.

To our knowledge, this is the **first application of DPO to a non-language, structured board-game domain** with discrete legal action constraints.

---

### **DPO vs SFT: Controlled Ablations**

We compare:

* Standard supervised fine-tuning (next-move prediction)
* Pairwise supervised fine-tuning
* Direct Preference Optimization (DPO)

under identical data, initialization, and evaluation conditions.

**Key findings:**

* DPO yields ~2× improvement in mean log-probability gap between the Grandmaster’s chosen move and Maia’s next-best alternative.
* DPO achieves these gains with **negligible additional KL divergence** from the base Maia-2 policy compared to the strongest SFT baseline.
* Preference optimization produces **stronger shallow-search performance** with less stylistic drift.

---

### **Quantitative Human-Likeness Evaluation**

We evaluate models on **held-out Grandmaster positions** (≈20%, stratified by game phase) using multiple complementary metrics:

* Top-1 accuracy on the Grandmaster move
* Mean log-probability gap (chosen vs rejected)
* KL divergence from the base Maia-2 policy
* Phase-wise behavioral statistics (opening / middlegame / endgame)
* Opening fingerprint similarity (ECO families)
* Tactical volatility and sacrifice propensity proxies

These metrics jointly measure **stylistic fidelity**, not just raw prediction accuracy.

---

### **Inference-Time Tactical Filtering with Engine Search**

Because Maia-2 operates below Grandmaster strength, we study whether tactical quality can be improved **without erasing learned style**.

At inference time only:

* Stockfish generates top-K candidate moves via MultiPV.
* Clear blunders are filtered using a centipawn-gap threshold relative to the best engine move.
* The learned policy (SFT or DPO) re-ranks the remaining candidates according to stylistic preference.

Importantly, **Stockfish is never used during training**—it acts purely as a tactical constraint during evaluation.

---

### **Strength vs Human-Likeness Tradeoff**

We perform a systematic **search-depth sweep**, varying Stockfish depth during inference:

```
Search depth ∈ {1, 2, 4, 8, 16}
```

For each depth, we measure:

* Engine-based Elo estimates
* All human-likeness metrics

This exposes a clear **Pareto frontier**:

* Increasing depth improves tactical strength
* Beyond a point, stylistic fidelity degrades as engine pressure dominates

DPO-trained policies retain stylistic alignment **longer under increasing depth** than SFT baselines.

---

## Experiments

### 1. Grandmaster-Likeness Evaluation

Held-out positions are used to measure how closely each policy aligns with the original Grandmaster’s decisions across phases and openings.

**Metrics include:**

* Top-1 move accuracy
* Log-probability gap statistics
* KL divergence vs base policy
* Opening distribution similarity
* Tactical volatility measures

---

### 2. Strength Estimation

Playing strength is estimated using a **local engine-vs-engine harness**:

* Fixed time controls
* Color swapping
* Elo estimates with confidence intervals

**Optional:** public evaluation via the official **Lichess Bot API** to provide an externally visible rating trajectory.

---

### 3. Depth Ablation (Key Insight)

By increasing tactical filtering strength at inference time, we isolate whether stylistic preferences learned via DPO persist even when decision-making is increasingly constrained by engine evaluation.

This experiment highlights where **human style survives** and where it breaks.

---

## Project Goals

Rather than pushing chess AI beyond human limits, **Grandmaster-DPO** focuses on:

* Faithfully modeling elite human decision-making
* Preserving stylistic diversity
* Enabling interpretability and historical analysis
* Supporting personalized, pedagogical chess AI

All fine-tuned models and evaluation code are released publicly to enable further research and community exploration.


## Repository Structure

```
grandmaster-dpo/
├── src/                # Training, inference, and evaluation code
├── scripts/            # Data prep, training, evaluation, deployment
├── configs/            # Model + training configs
├── data/               # Raw PGNs and metadata (not all included)
├── processed/          # Cleaned PGNs, train/val splits, JSONL
├── results/            # Evaluation outputs and logs
├── paper/              # Paper source, figures, bibliography
├── maia2_models/       # Base Maia2 checkpoints (if required)
├── checkpoints/        # (Ignored) trained model weights
├── LICENSE             # Apache 2.0 (code only)
├── NOTICE              # Model / artifact license clarification
├── CITATION.cff        # Citation metadata
└── README.md
```

---

## Reproducibility

This repository includes:

* Scripts for scraping and cleaning PGNs
* DPO and SFT training pipelines
* Evaluation harnesses for human-likeness and Elo
* Search wrappers for depth/beam ablations
* Optional bot deployment tooling

Exact commands are documented in `scripts/` and `configs/`.

---

[![Magnus style win vs Lichess AI Level 6](https://lichess.org/KuXD9to3.svg)](https://lichess.org/KuXD9to3)

<details>
<summary><strong>Maia Tuned on Magnus vs Stockfish level 6 PGN (click to expand)</strong></summary>


```pgn
[Event "casual correspondence game"]
[Site "https://lichess.org/KuXD9to3"]
[Date "2026.01.30"]
[Round "-"]
[White "Anonymous"]
[Black "lichess AI level 6"]
[Result "1-0"]
[GameId "KuXD9to3"]
[UTCDate "2026.01.30"]
[UTCTime "05:29:17"]
[WhiteElo "?"]
[BlackElo "?"]
[Variant "Standard"]
[TimeControl "-"]
[ECO "C25"]
[Opening "Vienna Game: Anderssen Defense"]
[Termination "Normal"]
[Annotator "lichess.org"]

1. e4 e5 2. Nc3 Bc5 { C25 Vienna Game: Anderssen Defense } 3. Nf3 d6 4. a3 Nf6 5. Be2 Nc6 6. b4 Bd4 7. Nxd4 Nxd4 8. O-O Be6 9. d3 a5 10. b5 h6 11. Rb1 O-O 12. f4 b6 13. Be3 Nxe2+ 14. Qxe2 Re8 15. Kh1 exf4 16. Rxf4 Qe7 17. Qf2 Ng4 18. Rxg4 Bxg4 19. Nd5 Qd7 20. Qg3 Kh7 21. Bd4 Rg8 22. a4 Rab8 23. h3 Be6 24. Nf4 c5 25. Bb2 c4 26. Qf3 g5 27. Nh5 Rg6 28. Rf1 Qc8 29. d4 Qh8 30. Bc3 g4 31. hxg4 Qg8 32. d5 Bxg4 33. Nf6+ Rxf6 34. Qxf6 Qg6 35. Qxf7+ Qxf7 36. Rxf7+ Kg6 37. Rf4 h5 38. Kh2 Rg8 39. Rf6+ Kh7 40. Rxd6 Bd1 41. Rd7+ Kh6 42. Bd2+ Kg6 43. Rd6+ Kg7 44. Rxb6 Kf7 45. Bxa5 Bxc2 46. Bc3 Rg6 47. Rxg6 Bxe4 48. Rf6+ Kg8 49. a5 Bxd5 50. b6 Be4 51. b7 Bxb7 52. a6 Be4 53. a7 Kh7 54. Rf4 Bd5 55. Rd4 Ba8 56. Rd7+ Kg6 57. Rd6+ Kg5 58. Rb6 Kh4 59. Bf6+ Kg4 60. Rb4 h4 61. Rb8 Be4 62. Rc8 Bd5 63. Rd8 Bc6 64. a8=Q Bxa8 65. Rxa8 h3 66. Rg8+ Kf5 67. Bc3 Ke4 68. Re8+ Kd3 69. Bf6 Kc2 70. gxh3 Kb3 71. Rc8 Kb4 72. Be7+ Kc3 73. h4 Kd4 74. Kg3 Kc3 75. Kf3 Kb3 76. h5 Kc3 77. h6 Kb3 78. h7 Kc2 79. Rxc4+ Kd3 80. h8=Q Kxc4 81. Qe5 Kd3 82. Qe4+ Kd2 83. Qe3+ Kd1 84. Qe2+ Kc1 85. Ke3 Kb1 86. Kd3 Ka1 87. Kc3 Kb1 88. Qb2# { White wins by checkmate. } 1-0
```
</details>

---

## Licensing

* **Code:** Apache License 2.0
* **Model weights and trained artifacts:** released separately under different terms

See `LICENSE` and `NOTICE` for details.

---

## References

* **AlphaZero**
* **Maia**
* **Maia2**
* **Direct Preference Optimization**

---

## Status

Active research project.
Paper and demo in progress.
