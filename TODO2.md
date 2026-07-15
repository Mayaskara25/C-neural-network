# TODO2 — edge-ml repo (Arduino ESP32 + on-device retraining)

This roadmap runs in the **`edge-ml`** repo (a copy of this project made *after* the `TODO.md`
work is done and green). Target: a **generic ESP32** dev board, **Arduino-ESP32** framework.

**Goal.** Run the trained ANN on-device for real-time sensor error correction. Once in a while
(~yearly, in the field) **retrain the model on the ESP32 itself** to compensate for sensor drift,
and **persist the updated weights to a flash partition** so they survive power cycles. This is the
thing TF-Lite Micro won't do — TFLM is inference-only.

**Hardware reality (generic ESP32, dual-core Xtensa LX6):**
- ~320 KB usable internal DRAM. **Do not assume PSRAM** (many generic boards lack it).
- **Hardware FPU for single-precision `float`.** `double` is **software-emulated → very slow**.
- Flash is limited and wears out (~100k erase cycles/sector); NVS does wear-leveling.
- Everything here must also run later on **ESP32-S3** (keep it portable; S3 just adds headroom
  and optional SIMD).

---

## Hard constraints to design around (read before coding)

1. **Heap fragmentation is the #1 killer.** The current design does `malloc`/`free` of matrices on
   *every* `forward_pass` and `backward_pass`. On a device that runs for months, the heap
   fragments until an allocation fails and the model crashes at 3 a.m. in a factory.
   → Move to **arena / static allocation**: preallocate every matrix once at init from one fixed
   buffer; forward/backward **reuse** them; **zero `malloc`/`free` in the steady-state loop.**
   This is the single biggest change and the core memory-management lesson of this repo.
2. **`float` only.** Audit for `double`. E.g. `sigmoid_activation` uses `1.0` (double) not `1.0f`
   at `src/activation.c:8`. Also **drop `powf`**: rewrite the sigmoid/tanh derivatives to their
   algebraic forms `s*(1-s)` and `1 - t²` — faster and no transcendental. (You may have already
   done the activation refactor in TODO.md; fold this in there.)
3. **No `printf`/`fprintf(stderr)` in the hot path.** Route to `Serial`, or wrap all training/
   progress prints (`print_bar`, the debug prints in `trainer_test.c`) in `#ifdef` so they compile
   out on-device.
4. **No filesystem by default.** Ship weights either baked into flash as a **C array**
   (xxd-style, inference-only) or in a **flash partition / NVS(Preferences)** so retraining can
   overwrite them.

---

## Phase 0 — Portability pass in the C core (do this on your PC first)

Get the code embedded-clean while you still have a debugger and valgrind.

- **Arena allocation path.** Add a mode that: (a) walks the network definition, (b) sums the bytes
  every matrix needs, (c) `malloc`s **one** arena, (d) hands out slices as `matrix` views into it
  (a matrix's `data` points into the arena; `free_mat` on a view must NOT free — see below).
  - Watch out: your `free_mat` currently `free()`s `data`. Arena views must not be individually
    freed. Either add an `owns_data` flag to `matrix`, or a separate `matrix_view` path, or simply
    never call `free_mat` on views and free the whole arena at once. Decide and be consistent —
    this is exactly the ownership discipline the project is about.
- **Keep the desktop malloc path under `#ifdef`** so you can still develop/test on PC.
- Purge `double`, purge `powf`, guard prints.
- **Verify `trainer_test` still trains to the same loss on desktop** after all this. If arena mode
  and malloc mode disagree, your arena slicing is wrong.

Run `valgrind` here — the last place you'll be able to.

---

## Phase 1 — Arduino library skeleton + inference-only demo

- Restructure as an Arduino library: `library.properties`, `src/` (Arduino compiles `.c` fine),
  `examples/inference/inference.ino`.
- **Bake a model into flash as a C array.** Train on desktop → `model.bin` → a small Python
  converter (put it in `scripts/`, next to the existing generators) emits `const uint8_t
  model_data[] = {...}`. Load that into the arena at `setup()`.
- `inference.ino`: feed synthetic sensor readings, run `predict()`, print corrected output over
  Serial.
- **Watch:** the Arduino `loop()` task has a fixed FreeRTOS stack (~8 KB). **Big matrices must live
  in the arena (heap), never on the stack** — a stack-allocated matrix will silently corrupt.
  Log `ESP.getFreeHeap()` at boot and after init to see your real headroom.

---

## Phase 2 — Persist the model to flash

- Store weights via **`Preferences`** (Arduino's NVS wrapper, `putBytes`/`getBytes`) or a dedicated
  **`esp_partition`** blob. Reuse your `save_model` byte layout (`src/trainer.c:157`) minus the
  `FILE*` — write the same bytes into an NVS blob instead.
- **Boot sequence:** try to load the model from flash → if absent or corrupt (check your
  `magic == 0xFAFF` and the version field you added in TODO.md) → **fall back to the baked-in
  factory model** from Phase 1. This gives the industry-friendly "factory default + field-updated"
  behavior: a bad flash write can never brick the device.
- **Only write flash after a retrain** (once a year), never per training step — respect the erase
  endurance. NVS handles wear-leveling; you handle *frequency*.

---

## Phase 3 — On-device retraining (the differentiator)

- **Keep gradient + optimizer buffers in the arena.** RAM cost of optimizer state on top of the
  weights: SGD = 0 extra, **momentum = 1× weights, Adam = 2× weights**. For the tiny nets this
  targets that's fine, but **default to plain SGD or momentum** to save RAM; make Adam opt-in.
- **Trigger:** a command over Serial/BLE, or a physical button, starts a calibration routine.
- **Routine:**
  1. Collect `N` fresh `(raw_sensor_reading → known_reference)` pairs into a small **fixed RAM
     buffer** (you know `N` up front → size it statically).
  2. **Mix in rehearsal data (see below)** so the batch is fresh-calibration + a few factory
     anchors, not fresh-only.
  3. Run a few epochs of SGD over that combined buffer (reuse `train`, but pointed at the RAM
     buffer and with prints compiled out). Low LR, few epochs.
  4. **Pass the validation gate (see below)** — only then overwrite flash.
  5. Keep the **factory model** as the ultimate fallback (Phase 2).

- **Rehearsal (anti-forgetting) — make it first-class, not an afterthought.** The failure mode
  here is *silent model degradation*, not a crash: a handful of calibration points + many epochs
  makes the net nail those points and drift everywhere else across the input range ("catastrophic
  forgetting"). Guard against it:
  - Bake a small **rehearsal set** — say 5–15 representative `(input → reference)` pairs spanning
    the sensor's full range — into flash alongside the factory model (fixed, never overwritten).
  - Every retrain trains on **fresh calibration pairs + the rehearsal set together**, so the model
    can't forget its baseline behavior while it corrects for drift.
  - Keep LR low and epochs few — you're *nudging* the model, not retraining from scratch.
  - RAM cost is tiny (a dozen small matrices) and it's the difference between "corrects drift" and
    "corrects drift at the calibration points and gets worse everywhere else."

- **Validation gate — first-class, blocks the flash write.** Never overwrite flash on faith:
  1. Keep a **held-out check set** in flash (distinct from the rehearsal set — points the retrain
     never sees) with known references.
  2. Measure error on it **before** retraining (current model) and **after** (candidate model).
  3. **Only commit the candidate to flash if `error_after < error_before`** by a real margin.
     Otherwise discard the candidate and keep running the old model. Log the before/after numbers
     over Serial so a field technician sees why a calibration was accepted or rejected.
  4. This single check is what makes on-device training *safe* to ship: a bad calibration (wrong
     reference, noisy readings, someone bumped the sensor) can never brick a working device.

- **Normalization under drift.** Your normalization params (`input_max`/`output_max`) are already
  saved with the model — persist and reload them. But drift can push new readings **past** the
  original max, so a normalized input exceeds ±1. Decide a policy: clamp, or refresh the max as
  part of calibration. Document whichever you pick.
- **Why this is even possible here and not in TFLM:** because you preallocated everything (Phase 0),
  retraining runs in a **fixed, known RAM budget with zero allocation churn**. That's the whole
  reason a 320 KB device can afford to train a small net in the field.

---

## Phase 4 — Generic-ESP32 vs ESP32-S3

- Keep everything **single-precision + arena-based** so it runs on a plain ESP32. Size the arena
  for **internal DRAM** (no PSRAM dependency).
- On S3 later: optionally add **SIMD / ESP-DSP** accelerated matmul, but always keep a **scalar
  fallback** compiled in so the same code still builds and runs on generic ESP32.

---

## Verification per phase
- **Desktop↔device parity:** same model + same input → same output within float tolerance
  (~1e-4). If they diverge, suspect a `double`→`float` change or an arena slicing bug.
- **Free-heap logging:** `ESP.getFreeHeap()` should be **flat** during the steady-state inference
  loop (proves zero allocation churn). A slow decline = a leak in the hot path — go find the
  stray `malloc`.
- **Retrain-improves-error assertion:** on a synthetic drift scenario (shift the sensor readings),
  assert post-retrain error < pre-retrain error on the held-out check set before the flash write
  is allowed (the validation gate from Phase 3).
- **Anti-forgetting assertion:** after a retrain, error on the **rehearsal set** must not have
  gotten meaningfully worse. If a single-drift-point calibration wrecks accuracy across the range,
  your rehearsal mix is too small or your LR/epochs too aggressive.
