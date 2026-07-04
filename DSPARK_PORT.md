# Adding DSpark speculative decoding to llama.cpp

Port of the **DSpark** draft (from [DeepSpec](https://github.com/edeleiter/DeepSpec))
into this fork, so a DSpark draft trained against `Qwen/Qwen3-4B` can accelerate a
GGUF target inside llama.cpp / LM Studio.

## Thesis: DSpark ≈ DFlash + a Markov-head logit bias

This fork already ships **DFlash** (`src/models/dflash.cpp`,
`common_speculative_impl_draft_dflash`, `LLM_ARCH_DFLASH`), a DeepSpec-family
block draft. DSpark reuses DFlash's machinery **unchanged**: multi-layer target
feature capture, `fc` fusion + `hidden_norm`, non-causal block drafting,
draft-side KV injection, verify/rewind, and the converter's target-metadata
plumbing. **No changes to `src/llama.cpp`, `src/llama-graph.cpp`, or the feature
extraction hooks.**

The one real delta is the **Markov head**: it adds a previous-token-conditioned
bias to the draft logits (`bias(prev) = markov_w2(markov_w1[prev])`), which makes
intra-block sampling **sequential** (token *i* depends on the token sampled at
*i-1*), where DFlash reads all block positions in one parallel pass.

**v1 scope:** VanillaMarkov head only; confidence head omitted (spec decoding is
lossless without it — the target is the sole arbiter); target = Qwen3-4B.

## Status

| Phase | Work | Status |
| --- | --- | --- |
| **A (Python)** | GGUF converter + arch/tensor/metadata registration | **Done + verified** |
| **A (C++)** | `LLM_ARCH_DSPARK`, hparams, `src/models/dspark.cpp`, factory | **Done + compiles + loads** |
| **B** | VanillaMarkov head in the speculative driver | **Done + compiles** |
| **C** | block layout / verification parity | Code done; run-test needs target |
| D | numeric validation vs DeepSpec `eval.py` `accept_len` | Gated on trained checkpoint |

The C++ (A(C++)-C) is written and compiles clean; a synthetic DSPARK GGUF loads
and allocates all tensors (incl. markov_w1/w2) - see "Done - C++" below. Numeric
validation (D) and the end-to-end draft run still need a checkpoint + a real
Qwen3-4B target, gated on DeepSpec Track 1.

## Done — Python converter (verified)

Files changed:
- `gguf-py/gguf/constants.py` — `MODEL_ARCH.DSPARK` (`"dspark"`), tensors
  `MARKOV_W1`/`MARKOV_W2` (canonical `markov_w1`/`markov_w2`), metadata keys
  `{arch}.markov_rank` / `{arch}.markov_head_type`, and the `DSPARK` tensor list
  (= DFlash's + the two markov tensors).
- `gguf-py/gguf/tensor_mapping.py` — `MARKOV_W1/W2` ← `model.markov_head.markov_w1/w2`.
  (`fc` ← `model.fc` and `hidden_norm` → `enc.output_norm` already exist from DFlash.)
- `gguf-py/gguf/gguf_writer.py` — `add_markov_rank`, `add_markov_head_type`.
- `conversion/qwen.py` — `DSparkModel(Qwen3Model)`, registered on
  `Qwen3DSparkModel` / `Qwen35DSparkModel`. Borrows the target tokenizer via
  `--target-model-dir`; writes `block_size`, `target_layers` (`[l+1 …]`, matching
  DeepSpec's `extract_context_feature`), `mask_token_id`, `markov_rank`,
  `markov_head_type`; **drops** `embed_tokens` / `lm_head` (borrowed from the
  target at load, like DFlash) and `confidence_head`.
- `conversion/__init__.py` — registry entries → `qwen`.

Verified via `import gguf`: the DSPARK arch resolves and `get_tensor_name_map`
maps `markov_w1/w2`, `fc`, `hidden_norm→enc.output_norm`, `norm`, and the attn/ffn
tensors correctly. (Full end-to-end conversion needs a checkpoint + torch.)

### GGUF contract (what the C++ side must read)

Metadata: `dspark.block_size` (u32), `dspark.target_layers` (array; the `+1`
layer ids), `dspark.markov_rank` (u32), `dspark.markov_head_type` (str),
`tokenizer.ggml.mask_token_id`. Tensors: the DFlash set (`fc`, `enc.output_norm`,
per-block `attn_*`/`ffn_*`, `output_norm`) **plus** `markov_w1` (`[rank, vocab]`,
get_rows) and `markov_w2` (`[rank, vocab]`, mul_mat). `tok_embd`/`output` are
borrowed from the target (not in the draft GGUF).

## Done — C++ (mirror DFlash)

Files changed (all compile clean; a synthetic DSPARK GGUF loads + allocates all
27 tensors incl. `markov_w1/w2`, then stops at the expected "requires ctx_other"
guard - proving arch resolution + tensor allocation):

- `src/llama-arch.{h,cpp}` - `LLM_ARCH_DSPARK` (`"dspark"`) + `LLM_TENSOR_MARKOV_W1/W2`
  (ops `GET_ROWS` / `MUL_MAT`). Reuses `LLM_KV_TARGET_LAYERS`, `FC`, `ENC_OUTPUT_NORM`.
- `src/llama-model.h` - `markov_w1`/`markov_w2` tensor fields + `uint32_t markov_rank`.
- `src/models/models.h`, `src/llama-model.cpp` - `llama_model_dspark` struct + factory /
  RoPE(NEOX) / `has_encoder` cases (clone of `llama_model_dflash`).
- `src/models/dspark.cpp` - clone of `dflash.cpp`; graph byte-for-byte identical (markov
  applied outside the graph). `load_arch_hparams` reads `dspark.markov_rank` via the
  loader's string-key overload; `load_arch_tensors` adds `markov_w1/w2` `{rank, n_vocab}`.
- `src/llama-context.cpp` - added DSPARK to the eagle3/dflash `ctx_other` (borrowed
  tok_embd/output) check.
- `src/llama-ext.h`, `src/llama-model.cpp` - accessors `llama_model_markov_w1/w2` (return
  the F32 `ggml_tensor*`) + `llama_model_markov_rank`.
- `common/common.h`, `common/speculative.cpp` - `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK`,
  registry `{"draft-dspark", ...}`, `need_n_rs_seq`/`n_max`/init-gate/dispatch, static_assert
  bumped 10->11, and `common_speculative_impl_draft_dspark` (clone of the dflash impl). The
  only logic delta is in `draft()`: after the single parallel block decode, the block is read
  **sequentially** - each position's base logits get `+= bias(prev)` in place via
  `llama_get_logits_ith`, then the untouched sampler runs; `prev` threads the sampled id.
- `tests/test-llama-archs.cpp` - DSPARK added to the eagle3/dflash smoke-test skips.

### Two decisions that diverge from the original sketch (per senior-architect review)

1. **Markov weights stay F32; bias is a plain CPU matvec in the driver** (no ad-hoc ggml
   graph). The converter pins `markov_w1/w2` to F32 (`tensor_force_quant`); the driver
   snapshots them to host once via `ggml_backend_tensor_get` (GPU-offload safe) and computes
   `bias(prev) = markov_w2 @ markov_w1[prev]` with a small loop. Deletes all
   quantized-backend risk. (Assumes the head is linear per the `bias(prev)=markov_w2(markov_w1[prev])`
   spec - if DeepSpec's VanillaMarkov has an activation/norm between w1 and w2, mirror it here;
   this is a Phase D validation item.)
2. **No new KV registrations** for scalars: `dspark.block_size` read via
   `llama_model_meta_val_str`, `dspark.markov_rank` via the loader string-key overload.

### Remaining (gated on a checkpoint + real target)

- End-to-end run: `llama-server --model qwen3-4b.gguf --model-draft dspark-*.gguf
  --spec-type draft-dspark`. Needs a real Qwen3-4B target (dspark borrows its
  tok_embd/output) and a DSpark checkpoint to convert.
- Numeric validation (Gate 1 losslessness, Gate 2 acceptance parity) - Phase D.
- Optional: add `draft-dspark` to `docs/speculative.md` (mirror the `draft-dflash` entries).

## Build & validate

```bash
cmake -B build && cmake --build build -j       # after the C++ lands

# convert a trained DSpark checkpoint (needs the target dir for the tokenizer/borrowed tensors):
python convert_hf_to_gguf.py <dspark_ckpt_dir> --target-model-dir <qwen3_4b_dir> --outfile dspark-qwen3-4b.gguf

# run:
./build/bin/llama-server --model qwen3-4b.gguf --model-draft dspark-qwen3-4b.gguf
```

**Gate 1 — losslessness (sampler-independent):** at temp 0, target+DSpark must
reproduce **target-only greedy** token-for-token. Proves correctness independent
of draft quality.

**Gate 2 — acceptance parity:** llama.cpp's `n_accept / n_drafted` matches DeepSpec
`eval.py`'s mean `accept_len` (run with `confidence_threshold=0` to match v1's
no-gate behavior) within ~5–10%.

**A/B tensor ladder** (build during the C++ phases, not after): replay identical
target features into both sides and compare, in order — extracted target-layer
inputs → fused `hidden_norm(fc(·))` → backbone hidden → base logits → markov bias →
corrected block — against the DeepSpec Python reference. First divergence localizes
the bug.

## References (DFlash template)

`src/models/dflash.cpp`; `common/speculative.cpp` (`common_speculative_impl_draft_dflash`
~903); `conversion/qwen.py` (`DFlashModel`); `src/llama-arch.cpp` (`LLM_ARCH_DFLASH`);
`gguf-py/gguf/constants.py` (`MODEL_ARCH.DFLASH`). DeepSpec source of truth:
`deepspec/modeling/dspark/qwen3/{modeling,config}.py`, `.../markov_head.py`,
`deepspec/eval/base_evaluator.py` (`generate_decoding_sample`).
