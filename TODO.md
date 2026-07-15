# TODO — Optimizers + Softmax

**Do all of this in THIS repo, and get it stable, BEFORE splitting into three repos.**
Everything here becomes the shared base that `legacy`, `sequence-models`, and `edge-ml` inherit.

Two independent pieces. Do **Part A first** (small, safe, teaches the core memory lesson), then
**Part B** (big refactor that touches every test file).

---

## Part 0 — Build a gradient checker FIRST (`tests/gradcheck.c`)

Before touching optimizers or softmax, build this. It is the **single most valuable debugging
tool** for a from-scratch NN, and every later part (Adam, softmax VJP, and especially LSTM/
transformer gradients in TODO1) leans on it. When something "trains but badly," gradient checking
tells you in seconds whether the bug is in the **math** or the **training loop** — otherwise
you're guessing.

**The idea:** compare your analytic (backprop) gradient against a numeric finite-difference
estimate. For one weight `w`, holding everything else fixed:
```
grad_numeric = ( loss(w + eps) - loss(w - eps) ) / (2 * eps)      // eps ≈ 1e-4
```
This should match `accumulated_dW` for that weight (remember your `/N` averaging — feed a single
example so N=1, or divide consistently). Relative error should be **< 1e-4**:
```
rel_err = |grad_analytic - grad_numeric| / (|grad_analytic| + |grad_numeric| + 1e-9)
```

**Sketch:**
1. Build a tiny net + one input/target pair. Run forward → backward → read `layer->dW`/`db`
   (analytic grads). **Do not** call the optimizer (don't mutate weights yet).
2. For each weight element: bump it `+eps`, forward, record loss; bump `-eps`, forward, record
   loss; restore the original value exactly. Compute `grad_numeric`.
3. Assert `rel_err < 1e-4` for every element. Print the worst offender.

**What it catches (all of which compile fine and look plausible):**
- transpose bugs (`dW = delta·xᵀ` vs `xᵀ·delta`),
- a mismatched sign or `1/N` factor between a loss and its derivative,
- a wrong softmax VJP or a bad fused-CE shortcut (Part B),
- later: BPTT gradient-accumulation bugs in the LSTM (TODO1).

Use `double`-free-friendly small `eps` and **single-precision awareness**: with float32, don't
expect better than ~1e-3–1e-4 relative error; a mismatch of 1e-1 means a real bug, 1e-4 is noise.
Seed `rand()` to a fixed value so a failure is reproducible.

Keep this test green after every change in Parts A and B.

---

## Part A — Optimizers: momentum, then Adam

### Where we are now
`update_weights_batch` (`src/model.c:79`) is plain SGD, applied directly to the averaged
accumulated gradients:

```
w -= lr * (accumulated_dW / batch_size)
```

There is **no per-parameter state** anywhere. Momentum and Adam need some.

### The one memory-management idea to internalize here

There are two *completely different* matrix lifecycles in this codebase, and mixing them up is
the classic bug:

| Lifecycle | Examples | Rule |
|-----------|----------|------|
| **Recomputed every step** | `dW`, `db`, `dx`, `delta`, `z`, `output` | freed + reallocated inside `backward_pass`/`forward_pass` every single call |
| **Persistent across steps** | `weights`, `bias`, `accumulated_dW`, and the NEW optimizer state | allocated **once**, reused, freed only at teardown |

Optimizer state (velocity, moments) is in the **persistent** bucket. If you `free_mat` +
`create_mat` the velocity each step, you wipe the momentum every step → the optimizer silently
degenerates back to SGD. **No crash, no leak — just wrong.** That's the lesson: correctness bugs
in C aren't always segfaults.

### Steps

**1. Add state fields to `DenseLayer` (`include/layer.h`).**
```c
// momentum
matrix v_dW, v_db;
// Adam (first + second moment)
matrix m_dW, m_db, v2_dW, v2_db;
```

**2. Init them to NULL in `create_dense_layer` (`src/layer.c:6`)** exactly like `accumulated_dW`
is done today (lines 17–18). Then **lazily allocate + zero on first use**, mirroring
`zero_accumulators` (`src/layer.c:71`):
```c
if (layer->v_dW.data == NULL) {
    layer->v_dW = create_mat(layer->weights.rows, layer->weights.cols);
    zero_mat(layer->v_dW);
    // ...same for v_db, and the Adam mats
}
```

**3. Free them in `free_dense_layer` (`src/layer.c:90`)** — add a `free_mat(&layer->v_dW);` line
per new field. **If you add a field and forget this line, that's your leak.** (Run under
`valgrind ./trainer_test` to confirm — see bottom.)

**4. Add the optimizer type (put it in `include/model.h`).**
```c
typedef enum { OPT_SGD = 0, OPT_MOMENTUM = 1, OPT_ADAM = 2 } OptimizerID;

typedef struct {
    OptimizerID type;
    float lr;
    float beta;      // momentum (0.9)
    float beta1;     // Adam (0.9)
    float beta2;     // Adam (0.999)
    float eps;       // Adam (1e-8)
    int   t;         // Adam timestep — SHARED across all layers, ++ once per update call
} Optimizer;
```

**5. Add `update_weights_optimizer(NeuralNetwork *nn, Optimizer *opt, int batch_size)`
(`src/model.c`).** Keep `update_weights_batch` around for legacy plain SGD. Branch on `opt->type`.
Do the math with **explicit `mat_at` loops** (like `update_weights_batch` already does at
`src/model.c:82-89`) — not chained `scalarmul_mat`/`sub_mat` calls. Reasons: no per-step
allocations, and it's the exact style you'll reuse on the ESP32 (TODO2).

The math (let `g = accumulated_dW[r][c] / batch_size`):

- **SGD:** `w -= lr * g`
- **Momentum:** `v = beta*v + g;  w -= lr * v`
- **Adam:** (increment `opt->t` once, at the very start of the call, before the layer loop)
  ```
  m  = beta1*m  + (1-beta1)*g
  v2 = beta2*v2 + (1-beta2)*g*g
  mhat = m  / (1 - beta1^t)
  vhat = v2 / (1 - beta2^t)
  w -= lr * mhat / (sqrtf(vhat) + eps)
  ```
  Use `powf(beta1, t)` for the bias correction. `g*g` is the elementwise square — just multiply
  the float, don't call `elemul_mat` (avoid the allocation).

Do the **same update for biases** using `accumulated_db` / `v_db` / `m_db` / etc.

**6. Wire it into `train` (`src/trainer.c:115`).** Add an `Optimizer *opt` parameter (or an
`OptimizerID` + build the struct inside). Replace the `update_weights_batch(...)` call at
`src/trainer.c:137` with `update_weights_optimizer(model->network, opt, b.size)`. Update the one
caller, `tests/trainer_test.c:46`.

### Order & how to check each step works
1. SGD path through the new function → should match old `update_weights_batch` behavior exactly.
2. Add momentum → on `xor_test` it should reach low loss in **fewer epochs** than plain SGD.
3. Add Adam → should be fastest and far less sensitive to `lr` (try lr=0.01 and lr=0.001; SGD
   would stall, Adam shouldn't).

### Optional but recommended: add a version field to the save format
`save_model` (`src/trainer.c:157`) writes `magic = 0xFAFF` then jumps straight to data — **no
version number.** Add one now:
```c
int version = 2;
fwrite(&magic, sizeof(int), 1, f);
fwrite(&version, sizeof(int), 1, f);   // NEW
```
and read+check it in `load_model_network` (`src/trainer.c:190`). Costs 4 bytes, and it means when
you later persist optimizer state (needed to *resume* on-device retraining in TODO2) old files
won't silently misparse. Bump to version 3 then, keep loaders for both.

> Note: optimizer state is **not** saved for now. Retraining just restarts the optimizer from
> zero, which is fine.

---

## Part B — Softmax via a vector-activation refactor

> Recommended approach (decided): make activations **vector-in / vector-out** so softmax is a
> first-class activation usable on any layer — this is the foundation the transformer work in
> TODO1 builds on. Keep **one** fused softmax+cross-entropy output path for numerical stability.

### Why the current design can't hold softmax

Activations today are **per-element**: `float (*activation)(float)`. `backward_pass`
(`src/layer.c:52-54`) does:
```c
act_deriv = other_op_mat(z, activation_derivative);   // elementwise f'(z)
delta     = elemul_mat(upstream, act_deriv);          // upstream ⊙ f'(z)
```
That `⊙` step is only valid because sigmoid/relu/tanh have a **diagonal Jacobian** (output `i`
depends only on input `i`). **Softmax is not diagonal** — every output depends on every input.

The saving grace: you never need the full Jacobian. The vector-Jacobian product is O(n):
```
s = softmax(z)
delta_i = s_i * ( upstream_i - dot(upstream, s) )
```

### The new activation interface

Replace the two scalar function pointers on `DenseLayer` with two **vector** ones:
```c
matrix (*activation_fn)(matrix z);                    // forward: z -> output
matrix (*activation_vjp)(matrix z, matrix upstream);  // backward: returns delta = J^T · upstream
```
Both **allocate and return** a matrix (caller frees) — same ownership convention as the rest of
the matrix API.

Existing scalar activations become thin wrappers (add to `src/activation.c`):
```c
matrix sigmoid_vec(matrix z)               { return other_op_mat(z, sigmoid_activation); }
matrix sigmoid_vjp(matrix z, matrix up) {
    matrix d = other_op_mat(z, sigmoid_derivative);
    matrix r = elemul_mat(up, d);
    free_mat(&d);
    return r;
}
```
(Same shape for relu/tanh/leaky/linear — this is literally the current behavior, relocated.)

Softmax:
```c
matrix softmax_vec(matrix z);              // subtract find_max(z) BEFORE expf, then normalize
matrix softmax_vjp(matrix z, matrix up);   // s = softmax_vec(z); delta_i = s_i*(up_i - dot(up,s))
```

### Blast radius — change these in THIS order (it won't compile mid-way, that's expected)

1. **`include/layer.h`** — change the two pointer field *types* (above); add `ActivationID act_id;`
   to the struct.
2. **`create_dense_layer` (`src/layer.c:6`) — change its signature to take an `ActivationID`
   instead of raw function pointers.** This is the keystone change. It fixes three things at once:
   - the he/xavier init check at `src/layer.c:22` (`activation == relu_activation`) — that pointer
     comparison is impossible once activations are vectors; instead `switch(act_id)`.
   - `get_activationId` (`src/trainer.c:16`) — no more fragile pointer→ID reverse mapping; just
     store `act_id` on the layer.
   - save/load — it writes `act_id` directly.

   Inside, resolve `act_id` → (`activation_fn`, `activation_vjp`) via the updated
   `get_activation_fxn`, and pick he vs xavier from `act_id`.
3. **`forward_pass` (`src/layer.c:47`)** → `layer->output = layer->activation_fn(layer->z);`
   (still `free_mat(&layer->output)` first).
4. **`backward_pass` (`src/layer.c:52-54`)** → replace the `act_deriv`+`elemul` block with:
   ```c
   free_mat(&layer->delta);
   layer->delta = layer->activation_vjp(layer->z, upstream_gradient);
   ```
   `dW`, `db`, `dx` below it are **unchanged**.
5. **`get_activationId` / `get_activation_fxn` (`src/trainer.c:16,24`)** — `get_activation_fxn`
   now outputs the two vector pointers; add a `case ACTIVATION_SOFTMAX`. `get_activationId` just
   returns `layer->act_id`. Add `ACTIVATION_SOFTMAX = 5` to the enum in `include/trainer.h:9`.
   **The save/load binary format does not change** — still one `int` actID per layer.
6. **All `create_dense_layer(...)` call sites** (~10 across `tests/*.c` — e.g.
   `tests/trainer_test.c:34-36`) → new signature, e.g.
   `create_dense_layer(1, 8, ACTIVATION_SIGMOID)`. Also update the load path at
   `src/trainer.c:229` which currently passes `activation, activation_derivative`.
7. **`src/activation.c` / `include/activation.h`** — add the vector wrappers + softmax fns.
8. **`src/loss.c` / `include/loss.h`** — add `LOSS_CROSS_ENTROPY` (categorical) and the fused
   path (below).

### The fused softmax + cross-entropy fast-path (do this for stability)

For classification, output layer = softmax, loss = categorical cross-entropy. The clean, stable
gradient collapses to:
```
delta_output = y_pred - y_actual
```
So in `train`, special-case it: **if** the output layer's `act_id == ACTIVATION_SOFTMAX` **and**
`loss_id == LOSS_CROSS_ENTROPY`, set the first delta to `y_pred - y_actual` directly and feed that
into `backward_network`, **skipping both** `loss_derivative_fn` **and** `softmax_vjp`. This avoids
the `1/(p·(1-p))`-style blowups. Keep `softmax_vjp` for the general case (softmax on a hidden
layer, and transformer attention in TODO1).

### Softmax gotchas
- **Overflow:** always subtract `find_max(z)` (`include/matrix.h:39`) before `expf`. Non-negotiable.
- **log(0):** clamp probabilities to `[eps, 1-eps]` in cross-entropy — reuse the exact pattern
  from `binary_cross_entropy` (`src/loss.c:120`).
- **Normalization:** `normalize_dataset` does per-feature max-abs. On one-hot targets the max-abs
  is 1, so it's a harmless no-op — but be aware you generally don't normalize classification
  targets.

### Test
Build a tiny 3-class toy set (or reframe xor as 2-class one-hot). Assert:
- softmax output columns sum to ~1.0,
- the fused CE path drives loss down and picks the right class.
- **Run `tests/gradcheck.c` (Part 0) against softmax_vjp AND the fused softmax+CE path.** The VJP
  is the easiest thing to get subtly wrong (the `- dot(up, s)` term). Check both the general path
  (softmax as a hidden activation) and the fused output path separately — they exercise different
  code.

---

## After both parts are green
Copy this repo into the two new repos and start `TODO1.md` (sequence-models) and `TODO2.md`
(edge-ml).

## Memory-check habit (do this after every part)
```bash
make trainer_test
valgrind --leak-check=full --error-exitcode=1 ./trainer_test
```
Zero leaks, zero invalid reads/writes. Every new `create_mat` must have exactly one matching
`free_mat`. This is the whole point of the project — treat a valgrind error like a failed test.
