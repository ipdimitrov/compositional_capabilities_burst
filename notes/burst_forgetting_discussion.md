# Burst Forgetting Mechanisms — Discussion Summary

## Setup

We study catastrophic forgetting caused by **data burstiness** using a synthetic bijection-composition task. A small transformer is:

1. **Pretrained** on background tasks: compositions of bijections across D=3 slots, each slot having 4 possible functions.
2. **Finetuned** with a mix of background + "burst" data, where one slot gets novel functions not seen during pretraining. The **burst fraction** (0–100%) controls how much of each batch is burst data.
3. **Forget phase**: trained on background-only data again, measuring how quickly burst knowledge is lost.

We sweep burst fraction and learning rate, tracking: accuracy, loss, gradient cosine similarity (burst vs bg), gradient norms, weight drift, SVD of weight deltas, etc.

## Key Observations

### 1. 100% burst forgets fastest — but not because of gradient conflict

At 100% burst, the model never sees background data during finetuning. Gradient cosine between burst and background is ~0 (meaningless — there's no bg signal in training). Yet it forgets the fastest by far.

**Explanation**: There's no bg gradient acting as a constraint, so the optimizer takes the shortest path to fit burst — but this overwrites features that bg relies on. During the forget phase, large bg gradients are needed to recover bg performance, and these corrections destroy burst knowledge.

### 2. Gradient cosine similarity is U-shaped across burst fractions

- **Low burst fracs (30–50%)**: cosine near 0 — model barely moves from pretrain, so burst and bg gradients are mostly orthogonal (independent), not opposed.
- **Mid burst fracs (70–85%)**: most negative cosine — the model is straddling the boundary between bg and burst basins. Both gradients are large and actively compete.
- **High burst fracs (95–100%)**: cosine drifts back toward 0 — model has left the bg basin entirely, so bg gradients point toward a distant optimum in high-dimensional space that isn't specifically anti-burst.

**Key insight**: Negative cosine requires both objectives to be "aware" of each other through shared parameters. This only happens in a transition zone. Too little burst = no conflict; too much burst = model has left the bg basin so bg gradients lose coherent directionality.

### 3. Mid-range burst fracs retain knowledge best (paradox of gradient conflict)

The fracs with the most negative gradient cosine (most conflict) during FT actually retain burst knowledge best during forgetting. This seems paradoxical but resolves as:

The bg gradient during FT acts as a **constraint / projection operator** — it prevents burst updates along bg-critical directions, forcing the optimizer to store burst knowledge in the **null space of the bg loss landscape**. More conflict during FT → burst stored more compatibly → bg training during forget doesn't erase it.

### 4. Weight drift tells a nuanced story

- 100% does NOT have the most weight drift during FT — it actually makes relatively small, efficient changes (no competing gradient to navigate around).
- But 100% has the MOST weight drift during forgetting — those small FT changes landed in critical shared directions, requiring large corrections to recover bg.
- Lower burst fracs: more drift during FT (longer path to satisfy both objectives), but less drift during forget (burst was already placed compatibly).

**Key insight**: It's not *how much* the weights change that matters, but *where*. Small changes in shared principal directions are more destructive than large changes in unused subspaces.

### 5. BG loss at end of FT is the best predictor of forgetting

The most monotonic relationship with forgetting severity is simply: **how much bg loss degraded during FT**. Higher bg loss at end of FT → stronger "recovery force" (large bg gradients) when forget starts → more burst knowledge destroyed.

This is directly monotonic with burst fraction and holds across learning rates. It's a simpler and more predictive metric than gradient cosine or weight drift alone.

### 6. Learning rate amplifies the effect

Higher LR during FT → more bg loss degradation → more forgetting. LR and burst fraction interact multiplicatively: high LR + high burst frac = maximal forgetting; low LR + low burst frac = maximal retention. The LR × concentration sweep heatmaps show this clearly.

### 7. SVD effective rank doesn't differentiate much

The SVD structure of weight deltas (effective rank, spectral norm) doesn't vary dramatically across burst fractions — it doesn't explain the forgetting differences we observe. The geometry of *which* directions change matters more than the rank/norm of the delta.

## Open questions

- Does making burst purely additive (identity slot during pretrain, new functions during FT) change the forgetting dynamics? Early experiments suggest the same pattern holds.
- Can we find a metric that captures "how much the weight changes overlap with bg-critical directions" more directly than bg loss?
- The U-shape in gradient cosine suggests an optimal burst fraction for knowledge retention — is this robust across architectures and tasks?

## Summary of the mechanism

Catastrophic forgetting from burstiness is driven by **where** weight updates land relative to the background loss landscape, not by gradient conflict per se:

1. High burst concentration → optimizer freely overwrites bg-critical features → bg loss rises → large recovery gradients during forget destroy burst knowledge
2. Lower burst concentration → bg gradient constrains updates to bg-compatible subspace → burst stored orthogonally → survives bg retraining
3. The gradient cosine measures direction conflict but misses the key factor: **bg loss displacement determines the magnitude of the recovery force during forgetting**
