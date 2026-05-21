# Related Work — Verifier-Bounded Iterative Self-Improvement (comprehensive)

*A complete, paper-grade survey of the surrounding literature, organized by line of
attack. Each cluster states what the prior work establishes and its **precise
limitation with respect to our question**: how accurate must an external verifier be
for **iterated self-consuming SFT** to keep improving rather than collapse, and is
that boundary a predictable function of model scale, family, and task?*

**Notation.** We denote verifier accuracy `acc_V` and decompose verifier error into a
two-parameter noisy channel — false-positive rate `ρ₀ = P(accept | wrong)` and
false-negative rate `ρ₁ = P(reject | correct)`. We deliberately avoid `α` for verifier
accuracy because the closest dynamics paper (2507.00075) already uses `α` for a
solver-decay coefficient; reusing it invites the misreading that we re-derive their law.

---

## 1. Self-improvement / self-training with a verifier

This cluster defines the **loop** we adopt — generate → verify/filter → fine-tune →
repeat — and supplies our protocol.

- **STaR** (Zelikman, Wu, Mu, Goodman 2022; arXiv 2203.14465). The original
  ground-truth-filtered iterated self-training: sample rationales, keep those that reach
  the correct answer (with rationalization for failures), fine-tune, repeat. Establishes
  the iterated-bootstrapping paradigm.
- **ReST** (Gulcehre et al. 2023; arXiv 2308.08998) and **ReST^EM "Beyond Human Data"**
  (Singh et al. 2023; arXiv 2312.06585, TMLR 2024). Formalize the grow/improve loop as an
  EM procedure and report the key empirical phenomenology we set out to characterize:
  profitable iterations are *few* (saturation by ~3 rounds), some tasks *regress* with more
  iterations (APPS at iteration 2), and gains *increase with model size*. Our loop protocol
  follows ReST^EM directly.
- **V-STaR** (Hosseini et al. 2024; arXiv 2402.06457). Trains a verifier as good as
  possible and uses it (plus DPO) — the "fixed, best-effort verifier" instantiation.
- **SPIN** (Chen et al. 2024; arXiv 2401.01335) and **Self-Rewarding LMs** (Yuan et al.
  2024; arXiv 2401.10020). Self-play / LLM-as-its-own-judge: the verifier signal is
  *internal* to the model. Self-Rewarding is also our motivating example of a real,
  *learned, imperfect* judge — the regime where `acc_V < 1` is native, not contrived.
- **Setlur et al.** (2024; arXiv 2406.14532, NeurIPS 2024) "RL on incorrect synthetic data
  scales reasoning 8×" — per-step credit on synthetic data; context for why *which*
  self-generated data is kept matters.

**Limitation (cluster 1).** The verifier is either a **near-perfect ground-truth checker**
(`acc_V ≈ 1`: answer match, unit tests) or the **model's own** judgment. None treats
verifier accuracy as an independent, continuously swept knob, so none can answer "how good
must the verifier be."

## 2. Reward-model overoptimization and controllable verifiers

The methodological precedent for *building a verifier whose quality you can dial*.

- **Gao, Schulman & Hilton** (2023; arXiv 2210.10760, ICML 2023) "Scaling Laws for Reward
  Model Overoptimization." The canonical controllable verifier: a *gold* RM defines truth,
  a *proxy* RM of tunable capacity stands in, and optimizing against the proxy yields a
  clean degradation law in KL distance. This gold/proxy construction is exactly the
  precedent for our controllable `acc_V`. **Limitation:** **one-shot** optimization against a
  static proxy — no iterated loop where this round's accepted labels become next round's
  training data, and no model-size axis.
- **RM-ensemble / robustness line:** Coste et al. (2023; 2310.02743, ensembles mitigate
  overoptimization), **WARM** (2401.12187), **WARP** (2406.16768), Setlur **PAV**
  (2410.08146), Multi-Agent Verification / BoN-MAV (2502.20379). These improve verifier
  *reliability* but treat the verifier as a fixed component, not a swept variable inside an
  iterated SFT loop.
- **Iterated RLHF overoptimization** (2505.18126) extends Gao to multiple rounds —
  closest in spirit to "iterate the overoptimization," but in RLHF/PPO with a reward model,
  not self-consuming SFT with a swept-accuracy verifier, and without a phase boundary in
  model size.

**Limitation (cluster 2).** Controllable-verifier methodology exists (Gao) and has even been
iterated (2505.18126), but always **one-shot or in RL**, never as a swept-accuracy verifier
governing an iterated self-consuming SFT loop with a model-size axis.

## 3. Model collapse and self-consuming dynamics

The cluster that studies *recursive self-training without an external verifier* — the
"what happens with no quality control" baseline our knob interpolates away from.

- **Shumailov et al.** (2024, *Nature* 631:755–759; arXiv 2305.17493 "Curse of Recursion").
  Canonical collapse-under-replacement: training on one's own generations narrows the
  distribution and degrades the tails. Our `replace` ablation reproduces its dynamics.
- **Gerstgrasser et al.** (2024; arXiv 2404.01413). *Accumulating* (rather than replacing)
  real+synthetic data bounds the error (σ²π²/6), so collapse is not inevitable. Our
  `accumulate` ablation reproduces this contrast.
- **Dohmatob et al.** (2024; arXiv 2402.07043, ICML 2024) "A Tale of Tails." Gives the
  iteration-modified scaling-law *template* `E^(n) ≍ T^{-c} + n·T₀^{-c}` (collapse iff
  `n ≫ (T₀/T)^c`) — the mathematical *form* our `acc_V`-dependent boundary generalizes, but
  in a data-size ratio with **no verifier**.
- **Dohmatob, Feng & Kempe** (2024; arXiv 2410.04840, ICLR 2025) "Strong Model Collapse."
  Has a **model-size × synthetic-fraction** Pareto and the nuanced "larger models can amplify
  or mitigate" claim — but the second axis is synthetic *fraction*, not verifier accuracy,
  and there is no mid-band and no fitted boundary in N.
- **Seddik et al.** (2024; arXiv 2404.05090). Statistical analysis of pure-recursion
  collapse. **Bertrand et al.** (2023; arXiv 2310.00429, ICLR 2024) "Stability of Iterative
  Retraining." Supplies the cleanest **proof technique** — a Jacobian/fixed-point phase
  threshold `λ(1+Lε/α) < 1/2` — but with no external verifier, convergence to the true
  optimum, and no model size.
- Generative-model siblings: **Self-Consuming Generative Models Go MAD** (2307.01850),
  **Chain of Diffusion / ReDiFine** (2407.17493), **latent-space-filtered self-consuming
  diffusion** (2511.12742) — same self-consumption pathology in image models.

**Limitation (cluster 3).** These either have **no verifier** (pure recursion) or use the
model's own likelihood; the central knob in our problem — an external verifier of swept
accuracy — is absent. They give us the collapse baseline, the math template (Dohmatob), and
the proof technique (Bertrand), but not the object we measure.

## 4. RL with noisy / imperfect verifiers (RLVR)

The most recent and most dangerous neighbor: papers that *do* sweep verifier quality, but
in **on-policy RL**, not self-consuming SFT.

- **Plesner et al.** (Apr 2026; arXiv 2604.07666) "An Imperfect Verifier is Good Enough."
  Sweeps verifier noise 0–15% across **three families (Qwen3, GLM4, Llama-3.1) at 4B–9B** in
  RLVR, concluding robustness below ~15% noise. The closest in *experimental shape*
  (multi-family × size × swept verifier noise).
- **Rad et al.** (Jan 2026; arXiv 2601.04411) "Rate or Fate? RLVεR." Uses **Youden's J =
  TPR − FPR** as the continuous verifier knob in GRPO and finds a sharp **J = 0** phase
  boundary — the cleanest existing "verifier-quality phase transition," but in RLVR.
- **Bao et al.** (Oct 2025; arXiv 2510.00915). Models verifier error explicitly as **FP/FN
  rates** in RLVR/GRPO — the same two-parameter channel we adopt, validating our `(ρ₀, ρ₁)`
  parameterization, but again in RL.
- **Dantas et al.** (Dec 2025; arXiv 2512.02080) "Designing Predictable LLM-Verifier
  Systems / the 4/δ bound." A Markov-chain analysis predicting iteration count `E[τ] ≤ 4/δ`,
  validated on 90k trials — but for **program verification** (formal proof loops), where
  verifier accuracy enters as a failure probability, not as a training-loop knob.

**Limitation (cluster 4).** All operate in **on-policy RL (GRPO/PPO-style)**, where there is
**no cross-round pseudo-label feed-forward**: a gradient is taken from a noisy reward, but
the accepted samples do not *become a frozen training set for the next round*. Self-consuming
SFT dynamics — this round's accepted labels are next round's data, so errors are *written
into the corpus* and compound — are categorically different. This is our single most
important setting modifier (see the RLVR+data-reuse cross-ablation in §10).

## 5. Self-improvement dynamics / solver–verifier gap / self-correction

Papers that *fit or prove a dynamics law* for self-improvement — but with an **internal**
verifier and (almost always) **no collapse branch**.

- **Song et al.** (2024; arXiv 2412.02674, ICLR 2025) "Mind the Gap." Studies the
  generation–verification gap and its scaling — but the verifier is the model's *own*
  self-consistency, and the law is over pretraining-FLOPs, not external verifier accuracy.
- **Huang et al.** (2024; arXiv 2412.01951, ICLR 2025 **Oral**) "The Sharpening Mechanism."
  Formalizes self-improvement as the model sharpening toward its own high-likelihood
  outputs — verifier = self-likelihood (internal). The canonical "internal verifier" contrast.
- **arXiv 2507.00075** (2025, ICLR 2026) "Solver-Verifier-Gap dynamics." **Our #1 dynamics
  threat.** A genuinely iterative retrain loop that **fits an exponential trajectory**
  `U_s(t) ≈ α'e^{−k(α−β)t} + U_{s,∞}` and reads off an ε-convergence time. Partially occupies
  the "fit iterative self-improvement dynamics, read off a time" framing. **Misses:** the
  verifier is the model's own self-verification (not external/swept), there is no model-size
  scaling axis, and the trajectory is smooth saturation with **no phase transition**.
- **arXiv 2602.10014** (2026) "Task-Centric Theory for Iterative Self-Improvement with
  Easy-to-Hard Curricula." Same iterative-SI framing, but the swept knob is **task
  difficulty**, not verifier accuracy, and it proves bounds rather than fitting a law.
- **Yang et al.** (2025; arXiv 2508.16456) "Probabilistic Inference Scaling Theory for LLM
  Self-Correction." A **closed-form whole-trajectory** prediction
  `Acc_t = Upp − αᵗ(Upp − Acc₀)` — the closest *formula* to ours. **Misses:** single-branch
  (monotone convergence, **no collapse arm**), and the setting is *inference-time
  self-correction*, not an iterated SFT retrain loop.
- **OpenReview X5Hk8aMs6w** (NeurIPS 2025) "Self-Verification Provably Prevents Model
  Collapse." The only neighbor with an explicit **collapse↔sustained boundary** — but via
  *self-*verification (internal), the boundary is in *verification sample size* (not model
  size), it is a **proven sufficient condition** (not a fitted empirical phase surface), and
  it is demonstrated on a toy GPT-2.
- **Weaver** (2506.18203, 2025) "Shrinking the Generation-Verification Gap with Weak
  Verifiers." Ensembles weak verifiers — but at **test time (best-of-K)**, with no training loop.

**Limitation (cluster 5).** The verifier is the model's **own self-verification** (internal,
not external/swept); the laws are over self-consistency or pretraining-FLOPs, not over an
external verifier's accuracy; and the trajectories are **monotone/saturating with no collapse
arm** (except X5Hk8aMs6w, which gives an existence proof, not a fitted surface).

## 6. The closest shadow: imperfect *external* verifier, but one-shot or unfitted

Two papers sit nearest to our exact object and must be addressed head-on.

- **Feng, Dohmatob, Yang, Charton & Kempe** (2024; arXiv 2406.07515, ICLR 2025) "Beyond
  Model Collapse." Parameterizes an imperfect **external** verifier `(ϕ, ψ)` and **proves a
  sharp one-shot phase threshold** `p⋆ = 1/(1+ψ/ϕ)`. This is the single closest neighbor on
  the "tunable external verifier + phase transition" pair. **Limitation:** **one-shot,
  N → ∞** — a single round with `N` = sample count, so it never sees the cross-round
  compounding that drives iterative collapse, and `N` is not model size. We are precisely its
  *iterated, finite-model* form.
- **Yi et al.** (2025; arXiv 2510.16657, v2 2026) "Escaping Model Collapse via Synthetic
  Data Verification." Iterative **self-consuming SFT** (our setting!) with an imperfect
  verifier modeled by **bias Δ + selectivity r**, and explicitly critiques Feng/Kempe's
  "sharp transition" as unrealistic (favoring a smooth one). **Limitation:** the verifier is
  parameterized by bias/selectivity rather than a 0→1 accuracy knob, there is **no
  model-size axis**, and the focus is the long-run convergence *point*, not a fitted
  iteration-count / phase surface in `(acc_V, N)` — they do not iterate-and-fit a boundary.

**Limitation (cluster 6).** The external-verifier + phase-transition idea exists, but only as
a **one-shot proof** (Feng/Kempe) or an **unfitted, size-free** convergence analysis (Yi).
Neither produces a fitted `(acc_V, N)` collapse↔sustained surface from an iterated loop.

## 7. Test-time scaling and inference compute

- **Brown et al.** (2024; arXiv 2407.21787) "Large Language Monkeys." Coverage law
  `c(k) ≈ exp(a·k^{-b})` over repeated samples. **Snell et al.** (2024; arXiv 2408.03314)
  optimal test-time compute. **Setlur et al.** (2025; arXiv 2502.12118, ICML 2025) "Scaling
  Test-Time Compute Without Verification or RL is Suboptimal" and inference scaling laws
  (2408.00724). **Limitation:** **no training loop** — these scale *inference*, not an
  iterated self-training process; the verifier (if any) ranks at decode time and is never
  written back into the model.

## 8. Theoretical templates and weak-supervision bounds we build on

- **Dohmatob 2024** (scaling-law form, §3), **Bertrand 2023** (fixed-point stability proof
  technique, §3) — borrowed for the functional form and the bifurcation argument behind our
  three-region motivating analysis.
- **Charikar, Pabbaraju, Hoffmann & Hashimoto** (2024; arXiv 2405.15116, NeurIPS 2024)
  "Quantifying the Gain in Weak-to-Strong Generalization" and **Wei, Shen, Saunshi & Ma**
  (2020; arXiv 2010.03622, ICLR 2021 Oral) "Theoretical Analysis of Self-Training." Bound how
  much a strong student can gain from weak/noisy supervision — theoretical siblings to "how
  much can a model learn from an `acc_V`-quality verifier," but static (no iteration, no fitted
  boundary).

---

## 9. Positioning matrix

Dimensions of the conjunction we claim (— absent, ~ partial, **+** present):

| Work | Iterated (T>1) | External verifier | Swept accuracy | Two-param FP/FN | Model-size axis | Fitted (acc_V,N) phase | SFT (vs RL) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| STaR / ReST^EM / V-STaR (cl. 1) | **+** | **+** | — (`acc_V≈1`) | — | ~ (gains rise w/ N, anecdotal) | — | **+** |
| SPIN / Self-Rewarding (cl. 1) | **+** | — (internal) | — | — | — | — | **+** |
| Gao 2023 RM-overopt (cl. 2) | — | **+** | ~ (RM size) | — | — | — | RL |
| Iterated RLHF overopt 2505.18126 | **+** | **+** | ~ | — | — | — | RL |
| Shumailov / Gerstgrasser / Dohmatob (cl. 3) | **+** | — | — | — | ~ (Strong-MC: synth frac) | — | **+** |
| Bertrand 2023 (cl. 3) | **+** | — | — | — | — | + (in ε,λ) | **+** |
| Plesner 2604.07666 (cl. 4) | **+** | **+** | **+** (0–15%) | — | **+** (4–9B) | — | **RL** |
| Rad 2601.04411 (cl. 4) | **+** | **+** | **+** (Youden J) | ~ (J=TPR−FPR) | — | + (sharp J=0) | **RL** |
| Bao 2510.00915 (cl. 4) | **+** | **+** | **+** | **+** (ρ₀,ρ₁) | — | — | **RL** |
| Song / Huang / 2507.00075 / Yang (cl. 5) | **+** | — (internal) | — | — | — | — (monotone, no collapse) | mixed |
| X5Hk8aMs6w (cl. 5) | **+** | ~ (self-verify) | — | — | — (verif. sample size) | ~ (proof, not fit) | **+** |
| **Feng/Kempe 2406.07515 (cl. 6)** ★ | **— (one-shot)** | **+** | **+** | ~ (ϕ,ψ) | — (N=sample count) | + (one-shot p⋆) | **+** |
| **Yi 2510.16657 (cl. 6)** | **+** | **+** | ~ (bias+selectivity) | ~ | — | — (convergence pt, unfitted) | **+** |
| **This proposal** | **+** | **+** | **+** | **+** | **+** (0.5–14B) | **+** | **+** |

No row but ours has **+** across all seven columns. The two closest (★ Feng/Kempe; Yi) each
miss the iteration-or-fit and the model-size axis; the RLVR cluster (Plesner/Rad/Bao) misses
the SFT feed-forward; the dynamics cluster (2507.00075/Yang) misses the external swept
verifier and the collapse branch.

## 10. The precise gap and one-clause delta vs each nearest neighbor

> **Gap.** No prior work measures a **continuously swept external verifier accuracy**
> (two-parameter `ρ₀, ρ₁`) inside **iterated self-consuming SFT**, across **model scale,
> family, and task**, and asks whether the collapse↔sustained boundary is a predictable
> function of `(acc_V, N)`.

One-clause delta per neighbor:

- **vs Gao 2023 / Feng-Kempe 2024** — we *iterate* (they are one-shot); our `N` is model size
  (theirs is sample count).
- **vs Mind-the-Gap / Sharpening / 2507.00075 / Yang 2508.16456** — our verifier is *external
  and swept* (theirs is the model's own), and we have a *collapse branch and a phase surface*
  (theirs are monotone/saturating).
- **vs X5Hk8aMs6w** — we give a *fitted empirical phase surface in model size*, not an
  existence proof in verification sample size.
- **vs Plesner / Rad / Bao (RLVR)** — we are *iterated self-consuming SFT* with pseudo-label
  feed-forward (they are on-policy RL with no cross-round corpus reuse).
- **vs Yi 2510.16657** — we sweep a 0→1 *accuracy* knob, add a *model-size axis*, and
  *iterate-and-fit* a boundary (they parameterize bias/selectivity, have no size axis, and
  characterize the long-run point).
- **vs ReST^EM / STaR** — we *sweep* `acc_V` instead of pinning it ≈1, and *fit* the law its
  data only hints at (few profitable iterations, gains rising with N).

## 11. Honest positioning of our contribution

- **Load-bearing novelty is the conjunction**, not any single ingredient. Each of {external
  verifier, swept accuracy, two-param FP/FN, iterated SFT, model-size axis, fitted phase
  surface} is individually occupied somewhere above; their **intersection is empty**. We
  frame nothing else as novel and cite the four closest threats
  (2507.00075, Feng/Kempe, X5Hk8aMs6w, Yi 2510.16657) plus the RLVR trio up front.
- **Experiment-layer vs analysis-layer.** The *measurement* — iterated self-consuming SFT
  trajectories indexed by `(ρ₀, ρ₁, N, family, task, dynamics)` under a standard protocol — is
  new regardless of outcome. The *three-region phase structure*, the *fitted boundary*, the
  *FP-vs-FN attribution*, and the *held-out predictability* are **analysis hypotheses assessed
  after the data** (the three-region structure and ≈0.85 held-out collapse prediction are so
  far observed only in a controlled tabular study, not in LLMs).
- **Not a Kaplan-2020 universal law.** The honest framing is a *controlled, predictable
  characterization* + a *mechanism* (FP-driven collapse) validated on real small LLMs, turning
  "how long can self-improvement safely run" from folklore into a measurable, falsifiable
  quantity in the iterated regime Gao/Feng-Kempe's one-shot analyses do not reach.
- **Reviewer-killer cross-ablation (planned).** To rule out "is the difference SFT-MLE vs PG,
  or data reuse?", a single key cell — RLVR **with** a frozen-replay buffer (data reuse) — is
  run; if it also collapses, "data reuse is the mechanism" sharpens the claim; if not,
  "SFT-MLE + reuse" is the specific locus. Either result empirically locks the setting modifier.
- **Risk.** The perimeter (self-improvement dynamics, verifier-bounded RL) is hot in 2024–2026
  (ICLR'25 Orals, ICML'25, NeurIPS'25, dedicated workshops; groups at NYU/Meta, MSR/Harvard,
  CMU, Stanford). The vacancy at the center is closing in months; **speed + the exact
  unoccupied conjunction are the moat.**

---

### References cited above (by arXiv id)

STaR 2203.14465 · ReST 2308.08998 · ReST^EM 2312.06585 · V-STaR 2402.06457 · SPIN 2401.01335 ·
Self-Rewarding 2401.10020 · Setlur RL-on-incorrect 2406.14532 · Gao RM-overopt 2210.10760 ·
Coste ensembles 2310.02743 · WARM 2401.12187 · WARP 2406.16768 · PAV 2410.08146 · BoN-MAV
2502.20379 · Iterated RLHF overopt 2505.18126 · Shumailov Nature/2305.17493 · Gerstgrasser
2404.01413 · Dohmatob Tale-of-Tails 2402.07043 · Strong Model Collapse 2410.04840 · Seddik
2404.05090 · Bertrand 2310.00429 · Self-Consuming MAD 2307.01850 · Chain-of-Diffusion
2407.17493 · Plesner 2604.07666 · Rad 2601.04411 · Bao 2510.00915 · Dantas 2512.02080 · Song
Mind-the-Gap 2412.02674 · Huang Sharpening 2412.01951 · Solver-Verifier-Gap 2507.00075 ·
Task-Centric 2602.10014 · Yang self-correction 2508.16456 · X5Hk8aMs6w (NeurIPS'25) · Weaver
2506.18203 · Feng/Kempe 2406.07515 · Yi 2510.16657 · Brown 2407.21787 · Snell 2408.03314 ·
Setlur VB-vs-VF 2502.12118 · Inference scaling 2408.00724 · Charikar W2S 2405.15116 · Wei
self-training 2010.03622.
