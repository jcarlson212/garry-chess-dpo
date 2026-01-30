Nice — yeah, you can absolutely do better than *Kasparov-Net*. That name undersells what you’re actually contributing.

Below is a **stronger framing + a rewritten README** that:

* Sounds like a *real ML paper*, not a fan project
* Scales beyond Kasparov (important for reviewers)
* Still allows you to *demo* “play like X” later
* Fits naturally with the repo name **`grandmaster-dpo`**

---

# **Grandmaster-DPO**

### *Preference Optimization for Human-Like Grandmaster Chess*

**Paper title (recommended):**

> **Preference Optimization for Human-Like Grandmaster Chess**

**Alternative paper titles (pick your vibe):**

* *Learning Human-Like Grandmaster Play via Direct Preference Optimization*
* *Beyond Imitation: Preference-Optimized Policies for Realistic Grandmaster Chess*
* *Human-Style Chess at Master Level via Preference Optimization and Shallow Search*

---

## Overview

**Grandmaster-DPO** is a research project studying whether **Direct Preference Optimization (DPO)** can be used to produce **strong, realistic, human-style chess play** from elite grandmaster data.

Starting from **Maia2**, a policy calibrated to human play at specific Elo bands, we apply DPO on curated grandmaster games (initially Garry Kasparov, with extensions to other GMs) and analyze how **preference learning reshapes the policy in ways that interact favorably with shallow search**.

Our focus is not on outperforming modern engines at high depth, but on identifying the **strength–human-likeness tradeoff frontier**: how far human-like policies can be pushed *without collapsing into engine-style play*.

---

## Research Questions

* Can preference optimization move a human-calibrated policy closer to **specific grandmaster playstyles**?
* Does DPO unlock **more effective shallow search** than supervised fine-tuning alone?
* How does search depth affect both **playing strength** and **behavioral fidelity**?
* Where does human-style play break under deeper tactical pressure?

---

## Core Contributions

* **Preference-based grandmaster imitation**
  We reframe grandmaster chess imitation as a preference-learning problem, using DPO rather than pure next-move prediction.

* **Quantitative human-likeness evaluation**
  We evaluate models on held-out grandmaster positions using:

  * Top-1 accuracy on the human move
  * Mean log-prob gap (chosen vs rejected)
  * KL divergence over legal moves (phase-wise)
  * Opening fingerprint similarity
  * Sacrifice / tactical volatility proxies

* **Strength vs human-likeness tradeoff analysis**
  By wrapping the policy in shallow search (varying depth and beam), we empirically map how:

  * Playing strength increases with depth
  * Behavioral fidelity degrades beyond a point
    This yields a clear **Pareto frontier** between realism and Elo.

* **DPO vs SFT ablations**
  We directly compare supervised fine-tuning and DPO under identical conditions, showing that DPO produces stronger shallow-search performance with less stylistic drift.

---

## Experiments

### 1. Grandmaster-Likeness Evaluation

Held-out GM positions (20%, stratified by opening / middlegame / endgame) are used to measure how closely the policy matches human decisions.

**Metrics:**

* Top-1 move accuracy
* Log-probability gap statistics
* KL divergence vs base policy
* Opening distribution similarity (ECO families)
* Sacrifice propensity and tactical volatility

---

### 2. Strength Estimation (Engine-Based Elo)

To avoid platform policy violations, we estimate strength using a **local engine-vs-engine harness**:

* Fixed time controls
* Multiple games with color swapping
* Elo + confidence intervals

**Optional:** public **Lichess bot evaluation**, using the official Bot API ecosystem, to provide an externally visible rating trajectory.

---

### 3. Depth Ablation (Key Insight)

We wrap the policy in a lightweight search and evaluate:

```
Search depth ∈ {1, 2, 4, 8, 16}
```

For each depth:

* Measure Elo
* Measure GM-likeness metrics

This experiment exposes the **human-likeness vs depth tradeoff** that engines typically hide.

---

### 4. Why DPO Instead of SFT?

A controlled comparison showing:

* Faster convergence
* Better shallow-search performance
* Lower KL drift for comparable strength gains

---

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
