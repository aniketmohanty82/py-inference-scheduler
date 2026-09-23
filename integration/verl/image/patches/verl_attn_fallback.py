"""Cherry-pick verl main's pure-torch bert_padding fallbacks into verl 0.9.0.

verl 0.9.0's _get_attention_functions() (utils/attention_utils.py) imports
flash_attn.bert_padding unconditionally on CUDA, and the PPO loop itself
reaches it: fit -> _compute_old_log_prob -> workers/utils/padding.py
left_right_2_no_padding -> unpad_input. No config avoids that call site -
attn_implementation, use_remove_padding and sequence parallelism all gate
OTHER imports of flash_attn, not this one - so on the vLLM 0.29.0 base
(torch 2.13.0+cu130, for which no flash-attn wheel exists on PyPI or in
Dao-AILab's releases) every arm dies at the first old-log-prob computation.

verl main already fixed this: the import is wrapped in try/except with pure
torch equivalents ("flash-attn only ships CUDA wheels, so CPU-only installs
... have to use the unoptimized but equivalent torch implementations"). This
patch transplants that block verbatim into the 0.9.0 file, so we stay on a
released verl instead of chasing main, which hard-pins vllm==0.24.0 and moves
daily. Engine-side generation is untouched (vLLM does its own attention);
this only affects the trainer's padding bookkeeping, identically in all
three arms.

Drop this patch when moving to the first verl release that contains the
fallbacks.

Exact-anchor replacement; fails loudly on drift.
"""

import os
import py_compile
import sys

import verl

TARGET = os.path.join(os.path.dirname(verl.__file__), "utils/attention_utils.py")

IMPORT_ANCHOR = (
    "        from flash_attn.bert_padding import "
    "index_first_axis, pad_input, rearrange, unpad_input"
)

IMPORT_REPL = """        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            # flash-attn only ships CUDA wheels, so CPU-only installs (unit tests, dev
            # boxes) have to use the unoptimized but equivalent torch implementations.
            index_first_axis = _fallback_index_first_axis
            pad_input = _fallback_pad_input
            rearrange = _fallback_rearrange
            unpad_input = _fallback_unpad_input"""

DEF_ANCHOR = "def _get_attention_functions()"

# Verbatim from verl main utils/attention_utils.py (fetched 2026-09-17).
FALLBACKS = '''
def _fallback_index_first_axis(tensor, indices):
    """Pure-torch equivalent of `flash_attn.bert_padding.index_first_axis`."""
    assert tensor.ndim >= 2
    return tensor[indices]


def _fallback_pad_input(hidden_states, indices, batch, seqlen):
    """Pure-torch equivalent of `flash_attn.bert_padding.pad_input`."""
    other_shape = hidden_states.shape[1:]
    output = hidden_states.new_zeros(batch * seqlen, *other_shape)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *other_shape)


def _fallback_unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Pure-torch equivalent of `flash_attn.bert_padding.unpad_input`."""
    import torch
    import torch.nn.functional as F

    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        _fallback_index_first_axis(hidden_states.reshape(-1, *hidden_states.shape[2:]), indices),
        indices,
        cu_seqlens,
        seqlens_in_batch.max().item(),
        used_seqlens_in_batch,
    )


def _fallback_rearrange(*args, **kwargs):
    """`einops.rearrange`, imported lazily so the other fallbacks stay einops-free."""
    from einops import rearrange as einops_rearrange

    return einops_rearrange(*args, **kwargs)


'''

src = open(TARGET).read()

if "_fallback_unpad_input" in src:
    print("verl_attn_fallback: already applied")
    sys.exit(0)

for name, anchor in (("import", IMPORT_ANCHOR), ("def", DEF_ANCHOR)):
    n = src.count(anchor)
    if n != 1:
        sys.exit(
            f"verl_attn_fallback: DRIFT - expected 1 {name} anchor, found {n} "
            f"in {TARGET}. Check whether the fallbacks landed in this verl."
        )

src = src.replace(IMPORT_ANCHOR, IMPORT_REPL, 1)
src = src.replace(DEF_ANCHOR, FALLBACKS + DEF_ANCHOR, 1)
open(TARGET, "w").write(src)
py_compile.compile(TARGET, doraise=True)

check = open(TARGET).read()
if check.count("_fallback_unpad_input") != 2:  # def + except-branch assignment
    sys.exit("verl_attn_fallback: post-check failed, wrong reference count")

print(f"verl_attn_fallback: patched {TARGET} (verl main's pure-torch fallbacks)")
