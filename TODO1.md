# TODO1 — sequence-models repo (LSTM → Transformers)

This roadmap runs in the **`sequence-models`** repo (a copy of this project made *after* the
`TODO.md` work — momentum/Adam + the vector-activation refactor — is done and green).

**Expectation-setting:** this is a large, multi-week, research-grade effort. LSTMs and especially
transformers are a real step beyond dense layers. Do it in the order below; get each phase to
overfit a tiny toy task before moving on. Don't skip straight to transformers.

**Prerequisite from TODO.md:** the vector activation interface (`activation_fn` / `activation_vjp`)
and softmax. Attention and layernorm are built on vector ops — without that refactor you'll be
fighting the framework.

---

## Phase 1 — Generalize the layer abstraction (memory lesson: a vtable in C)

Right now `DenseLayer` is a concrete struct and `NeuralNetwork` holds `DenseLayer**`
(`include/model.h:9`). You cannot bolt an LSTM onto that. Introduce polymorphism the C way — a
struct of function pointers plus an opaque `self`:

```c
typedef struct Layer {
    void *self;                                   // points to DenseLayer / LSTMLayer / ...
    matrix (*forward)(void *self, matrix input);
    matrix (*backward)(void *self, matrix upstream);   // returns dx
    void   (*update)(void *self, Optimizer *opt, int batch_size);
    void   (*zero_grad)(void *self);
    void   (*free)(void *self);
} Layer;
```

- `NeuralNetwork.layer` becomes `Layer**`. `forward_network` / `backward_network`
  (`src/model.c:41,49`) dispatch through the pointers instead of calling `forward_pass` directly.
- Wrap the existing dense code as the first `Layer` implementation (`DenseLayer` = the `self`).
- **Memory lesson:** you now own a two-level free — free each layer's `self` via its own `free`
  fn, then the `Layer` wrapper, then the array. One missed level = leak; freeing `self` twice =
  double-free. This is the pattern every "OOP in C" codebase (including the Linux kernel) uses.

Save/load will need a **layer-type tag** written per layer (dense vs lstm) so the loader knows
which constructor to call. Add `LayerType` to the enum family and bump the format version (you
added a version field in TODO.md Part A).

---

## Phase 2 — LSTM cell

An LSTM layer holds **four gate weight matrices** `W_f, W_i, W_c, W_o` (+ biases), each mixing the
concatenation of `[x_t, h_{t-1}]`. Per timestep you compute:
```
f_t = sigmoid(W_f · [x_t, h_{t-1}] + b_f)     (forget gate)
i_t = sigmoid(W_i · [...] + b_i)              (input gate)
g_t = tanh   (W_c · [...] + b_c)              (candidate)
o_t = sigmoid(W_o · [...] + b_o)              (output gate)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```
All of sigmoid/tanh/elemul already exist. You'll also need a **concatenate** op for `[x_t, h_{t-1}]`.

### The memory challenge: BPTT needs every timestep's activations
Backprop-through-time walks the sequence in reverse, and the gradient at step `t` depends on the
gates/cell-state you computed at step `t` in the forward pass. So you must **cache all timesteps**:
`f_t, i_t, g_t, o_t, c_t, h_t` for every `t`. For a length-`T` sequence that's `~6*T` matrices.

- Store them in an **arena** (one big preallocated block, or an array-of-matrices indexed by `t`),
  not ad-hoc mallocs in the loop.
- **Forward:** iterate `t = 0..T-1`, carry `h_t`, `c_t`, stash each step's activations.
- **Backward (BPTT):** iterate `t = T-1..0`. The four weight matrices are **shared across all
  timesteps**, so their gradients **accumulate** over `t` — you add into the same `dW_f` etc.,
  you do **not** overwrite. This is the *exact* persistent-accumulator pattern from the optimizer
  work in TODO.md — same mental model, reused.
- Carry two gradients backward across time: `dh_{t-1}` and `dc_{t-1}`.

### Gradient clipping (required for LSTMs)
LSTMs explode without it. Before the optimizer step, compute the global grad norm and scale all
gradients down if it exceeds a threshold (clip-by-norm, threshold ~1.0–5.0). Add this as a step
inside `update` or just before it.

### Test
A sequence memorization task: echo (output = input delayed by k steps), or predict-next in a
simple periodic sequence. If it can't overfit a handful of short sequences, the BPTT gradient
accumulation is wrong — check that weight grads accumulate rather than overwrite.

---

## Phase 3 — Transformer blocks (only after LSTM works)

Build these ops in order; each should overfit a toy copy task before you stack them.

1. **Scaled dot-product attention (single head).**
   `attn(Q,K,V) = softmax(Q·Kᵀ / sqrt(d_k)) · V`. The softmax is over the **sequence axis** — you
   need an **axis-aware softmax** (your TODO.md `softmax_vec` softmaxes a whole vector; here you
   softmax each row/column of a matrix). Write that first.
2. **LayerNorm (new op, needs forward AND backward).** Normalize over the **feature dim**:
   `y = γ · (x - μ)/sqrt(σ² + eps) + β`, with learned `γ, β`. The backward is the fiddly one —
   derive it carefully (mean and variance both depend on every element).
3. **Residual + FFN.** Residual = matrix add (exists). FFN = two dense layers with GELU/ReLU
   between (dense exists; add GELU as a vector activation per TODO.md's interface).
4. **Assemble a transformer block:** `x → +attn(LN(x)) → +FFN(LN(x))` (pre-norm), then stack.
5. **Multi-head:** split features into heads, attend per head, concat, project. Mostly reshaping
   and bookkeeping over the 2D matrix.
6. **Positional encoding:** add sinusoidal (or learned) position signal to the input embeddings.

### Matrix gaps to fill before you start
- **Axis-aware softmax** (row-wise / col-wise) with its VJP.
- **LayerNorm forward + backward.**
- Your `matrix` is already 2D row-major, which maps cleanly onto **sequence × features** — good,
  no data-layout rewrite needed.

Get a **single attention block to overfit a tiny copy task** (output = input) before scaling to
multiple heads/layers. If one block can't memorize, more blocks won't help.

---

## Recurring memory-management themes in this repo
- **vtable generic layer** (function pointers + `void *self`) — two-level ownership/free.
- **Per-timestep activation caches** for BPTT — arena allocation, indexed by `t`.
- **Gradient accumulation across time** — shared weights, add-don't-overwrite.
- Eventually a **scratch/arena allocator** to stop the malloc-per-matrix-op churn (attention
  allocates a lot of temporaries) — this is also the bridge to how `edge-ml` (TODO2) works.
