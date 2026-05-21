# Verifier-Bounded Iterative Self-Improvement: When Does Self-Training Collapse, and Can We Predict It?

*Comprehensive proposal / paper draft. Experiment-layer claims (the controlled measurement) are
established-by-design; analysis-layer claims (phase structure, FP/FN asymmetry, predictive law) are
hypotheses to be assessed after the data is collected.*

---

## Abstract

Large language models increasingly improve themselves by generating their own training data,
filtering it with a *verifier*, and fine-tuning on what survives (STaR, ReST^EM, RLAIF, o1/R1-style
pipelines). In every realistic setting the verifier is **imperfect** — a learned reward model or
LLM-judge, not a ground-truth oracle. Yet no prior work systematically asks the central question:
**how accurate must the verifier be for iterative self-training to keep improving rather than
collapse, and is that boundary predictable?** We treat verifier accuracy as a *controlled,
continuously swept* variable inside a standard iterated self-consuming SFT loop, and measure
self-training trajectories across a two-parameter verifier (false-positive rate ρ₀, false-negative
rate ρ₁), model scale, model family, and task. The resulting measurement — controlled verifier
accuracy × iterated SFT × scale/family/task — does not exist in the literature. Our analysis then
asks whether a predictive phase boundary emerges; we report the upper bound (a universal predictive
law) and the lower bound (a first systematic empirical characterization, possibly a
heterogeneity/negative result) the experiment can deliver.

## 1. Introduction

Self-improvement — a model bootstrapping its own capability from self-generated data — underpins
much recent progress (STaR, ReST^EM, Self-Rewarding LMs, RLAIF, and reasoning-RL systems). The
mechanism is a loop: **generate → verify/filter → train → repeat**. Crucially, the model trains on
its *own* generations, not on external human-written solutions; the only external grounding is the
verifier that decides which self-generations are good enough to learn from.

Two facts make verifier quality the pivotal variable. First, with **no** verifier (or a verifier
equal to the model's own likelihood), recursive self-training is known to **collapse** — errors
compound and the distribution narrows (model collapse). Second, in the regimes that matter most in
practice — open-ended generation, reasoning without checkable answers, RLAIF — the verifier is a
**learned, imperfect** model, not a ground-truth checker. Self-improvement therefore lives or dies
by *how good the imperfect external verifier is*. Despite this, the field has not characterized the
relationship: **how imperfect can the verifier be before iterative self-training stops helping and
starts hurting, and does this boundary follow a predictable law across model scale and family?**

Our contribution separates cleanly into two layers:

- **Experiment (novelty is outcome-independent).** We produce the first systematic measurement of
  iterated self-consuming SFT trajectories under a *controlled, swept external verifier accuracy*,
  across model scale (Qwen2.5 0.5B–14B), six model families, and two tasks. This controlled
  measurement is a new empirical object regardless of what it reveals.
- **Analysis (contribution is conditional on the data).** Given the trajectories, we ask: is there
  a phase transition? a fittable boundary acc_V*(N)? is it predictable on held-out settings?
  universal across families/tasks? is collapse driven by false-accepts rather than false-rejects?

## 2. Related Work and Their Limitations

We organize prior work by line of attack; each has a specific limitation **with respect to our
question** (the bracketed gap).

**(a) Self-training / self-improvement with a verifier.** STaR (Zelikman et al. 2022),
ReST (Gulcehre et al. 2023), ReST^EM "Beyond Human Data" (Singh et al. 2023), V-STaR (Hosseini
et al. 2024), SPIN (Chen et al. 2024), Self-Rewarding LMs (Yuan et al. 2024). These define the
iterated loop we adopt. *Limitation:* the verifier is either a **near-perfect ground-truth checker**
(answer match / unit tests, α≈1) or the **model's own** judgment; none treats verifier accuracy as a
swept variable, so they cannot say how good the verifier must be.

**(b) Reward-model overoptimization and verification.** Gao, Schulman & Hilton (2023, "Scaling Laws
for Reward Model Overoptimization") give the canonical *controllable* verifier via a gold/proxy RM
split and fit a clean degradation law in KL; Feng & Kempe et al. (2024, "Beyond Model Collapse")
parameterize an imperfect verifier (ϕ,ψ) and prove a one-shot phase threshold p⋆ = 1/(1+ψ/ϕ).
*Limitation:* both are **one-shot** (single round / N→∞), so they never see the cross-round
compounding that drives iterative collapse, and they have no model-scale axis.

**(c) Model collapse.** Shumailov et al. (2024, Nature), Gerstgrasser et al. (2024, accumulate vs
replace), Dohmatob et al. (2024, "Tale of Tails" — iteration-modified scaling law), Seddik et al.
(2024), Bertrand et al. (2023, iterative-retraining stability). These study recursive self-training
dynamics and collapse. *Limitation:* they have **no verifier** (pure recursion) or use the model's
own likelihood; the central knob in our problem is absent.

**(d) RL with noisy/imperfect verifiers (RLVR).** Plesner et al. (2026, "An Imperfect Verifier is
Good Enough"; sweeps verifier noise 0–15% across families/sizes), Rad et al. (2026, "Rate or Fate?";
Youden's J = TPR−FPR knob, sharp J=0 boundary), Bao et al. (2025; FP/FN noise channel). These sweep
verifier quality. *Limitation:* they operate in **on-policy RL (GRPO-style)**, where there is no
cross-round pseudo-label feed-forward; the self-consuming SFT dynamics — where this round's accepted
labels become next round's training data — are different.

**(e) Self-improvement dynamics / solver-verifier gap.** Song et al. (2024, "Mind the Gap"), Huang
et al. (2024, "Sharpening Mechanism", ICLR'25 Oral), arXiv 2507.00075 (solver-verifier-gap dynamics;
fits an exponential trajectory and an ε-convergence time), Yang et al. (2508.16456, a closed-form
self-correction trajectory Acc_t = Upp − αᵗ(Upp − Acc₀)). *Limitation:* the verifier is the model's
**own self-verification** (internal, not external/swept); the laws are over self-consistency or
pretraining-FLOPs, not over an external verifier's accuracy; trajectories are **monotone/saturating
with no collapse arm**.

**(f) Test-time scaling.** Brown et al. (2024, "Large Language Monkeys"), Snell et al. (2024,
test-time compute). *Limitation:* **no training loop** — inference-time only.

**(g) How the field models a controllable verifier.** Gao's gold/proxy split (verifier quality =
proxy-RM capacity), Feng/Kempe's (ϕ,ψ), Bao's two-rate channel P(R̃=1|R*=0)=ρ₀, P(R̃=0|R*=1)=ρ₁,
Rad's Youden's J. The field's standard is a **two-parameter** (FP, FN) channel; a single symmetric
accuracy is the weakest variant. We adopt the two-parameter form.

**Synthesis.** Every line fixes one dimension: perfect verifier (a), one-shot (b), no verifier (c),
RL-not-SFT (d), internal self-verification (e), no training (f). The combination at the center is
unoccupied.

## 3. The Gap

> No prior work measures **a continuously swept external verifier accuracy** (two-parameter ρ₀, ρ₁)
> inside **iterated self-consuming SFT**, across **model scale, family, and task**, and asks whether
> the collapse↔sustain boundary is predictable.

This is the precise conjunction we bridge. It is distinct from one-shot RM overoptimization (b),
from verifier-free collapse (c), from on-policy RLVR (d), and from internal self-verification (e).

## 4. Our Approach

### 4.1 Loop and protocol (standard, prior-work-anchored)
We use the ReST^EM loop: each round the current model generates K candidate solutions per training
prompt; the verifier accepts/rejects each; the model is LoRA-fine-tuned on the accepted (prompt,
self-generated-solution) pairs; repeat. Protocol parameters follow ReST^EM/STaR/2507.00075:
- training prompts: a fixed standard subset (≈2048, per 2507.00075) up to the full GSM8K train set;
- K = 16 samples/prompt at temperature 1.0; cap ≤10 accepted solutions/problem (ReST^EM);
- **restart-from-base + replace** as the primary dynamics (ReST^EM/STaR); continue/accumulate as ablations;
- LoRA (r=16, attn+MLP) + AdamW, lr 1e-4, 2 epochs/round;
- evaluation on the **full** held-out test set, greedy pass@1 (+ pass@k);
- **≥3 seeds** with reported variance (exceeds the field's typical n=1).

### 4.2 The controllable verifier (the knob)
Verifier modeled as a two-parameter noisy channel on the accept/reject decision relative to gold:
**ρ₀ = P(accept | solution is wrong)** (false-positive; injects a wrong label into next round's
training set) and **ρ₁ = P(reject | solution is correct)** (false-negative; discards a good example).
Gold answers (GSM8K final answer / ARC option) are used **only** to define the verifier and to
evaluate — never as training targets. Main sweep: symmetric ρ₀=ρ₁=ρ, verifier accuracy α=1−ρ ∈
{1.0,…,0.4}. **Asymmetric slices** (ρ₀-only vs ρ₁-only) test whether collapse is FP- or FN-driven.

### 4.3 Axes
verifier accuracy α (and the FP/FN decomposition) × model scale N (Qwen2.5 0.5/1.5/3/7/14B) × model
family (Qwen, Llama-3.1-8B, Gemma-2-9B, OLMo-2-7B, DeepSeek-R1-Distill-Qwen-7B, Phi-3.5-mini) × task
(GSM8K, ARC-Challenge); dynamics ablation (restart/continue × replace/accumulate).

### 4.4 Infrastructure
vLLM for high-throughput generation (one-shot subprocess per round, clean GPU release) + PEFT-LoRA
for training; orchestrated as an iterated loop. Each (α, model, task, seed) cell yields a full
per-round test-accuracy trajectory.

### 4.5 What the experiment produces (novelty, outcome-independent)
A dataset of iterated self-training trajectories indexed by (controlled verifier accuracy ρ₀/ρ₁,
scale, family, task, dynamics) under a standard protocol — a measurement that does not currently
exist. Its value does not depend on the analysis conclusion.

## 5. Hypotheses and Expected Results

Pre-registered analysis hypotheses (assessed after data; not assumed):
- **H1 (existence):** there is a critical verifier accuracy α*(N) separating sustained improvement
  from collapse.
- **H2 (form):** the profitable-iteration count / boundary is a regular function of (α, N).
- **H3 (predictability):** a model fit on all but one held-out setting predicts its collapse/sustain
  class.
- **H4 (universality):** the boundary is consistent across families/tasks (vs. model-specific).
- **H5 (mechanism):** collapse is driven by false-positives (ρ₀), not false-negatives (ρ₁).

**Upper bound (best case):** a clean, fittable, **universal and predictable** phase boundary
α*(N) — a scaling-law-like predictive law for verifier-bounded self-improvement, with the FP/FN
decomposition isolating the causal driver. Field-relevant (COLM/NeurIPS/ICLR-class).

**Lower bound (worst / null):** no clean universal law — heterogeneous, family-specific behavior.
Even then the experiment delivers the **first systematic measurement** of verifier-accuracy effects
on iterated self-training and a publishable, practically useful finding ("collapse susceptibility is
not predictable from scale/competence alone"). Non-zero (workshop / empirical-study tier).

**Preliminary motivation (controlled, not the LLM result).** In a controlled tabular study
(logistic-regression/MLP on synthetic + real data, no LLM), we observe a three-region structure
(collapse / mid-band / converged) and a held-out collapse-vs-sustain prediction accuracy of ≈0.85.
This motivates the LLM measurement; whether the same structure and predictability hold at LLM scale
is exactly what the main experiment tests.

## 6. Positioning and Honest Assessment

- **Experiment-layer novelty is certain**: the controlled (verifier-accuracy × iterated-SFT ×
  scale/family/task) measurement is new.
- **Analysis-layer contribution is conditional**: a predictive universal law (upper bound) vs. an
  empirical characterization / heterogeneity result (lower bound).
- **Not** a universal scaling law in the Kaplan-2020 sense; the honest framing is a controlled,
  predictable characterization plus mechanism (FP-driven collapse), validated on real small LLMs.
- **Risk:** the surrounding area (self-improvement dynamics, verifier-bounded RL) is active in
  2024–2026; speed and the precise unoccupied conjunction are the moat.

## References (by role)

Self-training: Zelikman 2022 (STaR, 2203.14465); Gulcehre 2023 (ReST, 2308.08998); Singh 2023
(ReST^EM, 2312.06585); Hosseini 2024 (V-STaR, 2402.06457); Chen 2024 (SPIN, 2401.01335); Yuan 2024
(Self-Rewarding, 2401.10020). Verification/overoptimization: Gao 2023 (2210.10760); Feng/Kempe 2024
(2406.07515). Model collapse: Shumailov 2024 (Nature; 2305.17493); Gerstgrasser 2024 (2404.01413);
Dohmatob 2024 (2402.07043); Seddik 2024 (2404.05090); Bertrand 2023 (2310.00429). RLVR-noise:
Plesner 2026 (2604.07666); Rad 2026 (2601.04411); Bao 2025 (2510.00915). Self-improvement dynamics:
Song 2024 (2412.02674); Huang 2024 (2412.01951); 2507.00075; Yang 2025 (2508.16456). Test-time:
Brown 2024 (2407.21787); Snell 2024 (2408.03314).
