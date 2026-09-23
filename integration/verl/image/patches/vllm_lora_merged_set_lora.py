"""Cherry-pick vLLM PR #51423: normalize fused lora_a/lora_b in merged set_lora.

Without it, vLLM 0.29.0 cannot mount verl's LoRA adapter and every arm dies
before step 1 with:

    vllm/lora/layers/column_parallel_linear.py:261 in slice_lora_b
    IndexError: too many indices for tensor of dimension 1

MergedColumnParallelLinearWithLoRA.set_lora() declares both weights as
`Tensor | list[Tensor]` but normalizes neither into per-slice lists. A fused
source hands over one lora_a shared by every slice and one lora_b covering all
of them; indexing either as if it were a per-slice list yields a 1-D row. The
PR reports 12 of 16 fused/sharded combinations failing at tp=2, which is our
topology.

Upstream is open and unmerged as of 2026-09-17 (filed 2026-08-07, fixes
#51409), so there is no release to upgrade to. This is an ENGINE fix, applied
identically to all three arms, so it cannot bias the store-vs-stock
comparison - the same reasoning that governs the connector_v3 watchdog.

Drop this patch once #51423 lands in a release.

Exact-anchor insertion; fails loudly on drift.
"""

import os
import py_compile
import sys

import vllm

TARGET = os.path.join(
    os.path.dirname(vllm.__file__),
    "lora/layers/column_parallel_linear.py",
)

# Context line the upstream hunk is inserted directly above.
ANCHOR = "        # Expand packed adapter groups when they don't match n_slices."

INSERT = '''        # A fused source (e.g. Megatron, which stores QKV as one linear) hands
        # over a single lora_a shared by every slice and a single lora_b
        # covering all of them. Split b by output_sizes and replicate a -- the
        # same normalization MergedColumnParallelLinearVariableSliceWithLoRA
        # already applies. Normalize b first so the a count below matches.
        if isinstance(lora_b, torch.Tensor):
            split_b, start = [], 0
            for size in self.output_sizes:
                split_b.append(lora_b[start : start + size])
                start += size
            lora_b = split_b
        if isinstance(lora_a, torch.Tensor):
            n = len(lora_b) if isinstance(lora_b, list) else self.n_slices
            lora_a = [lora_a] * n

'''

src = open(TARGET).read()

if "Normalize b first so the a count below matches." in src:
    print("vllm_lora_merged_set_lora: already applied")
    sys.exit(0)

n = src.count(ANCHOR)
if n != 1:
    sys.exit(
        f"vllm_lora_merged_set_lora: DRIFT - expected 1 anchor, found {n} in {TARGET}. "
        f"Check whether #51423 landed upstream and drop this patch."
    )
if "import torch" not in src:
    sys.exit(f"vllm_lora_merged_set_lora: DRIFT - {TARGET} does not import torch")

src = src.replace(ANCHOR, INSERT + ANCHOR, 1)
open(TARGET, "w").write(src)
py_compile.compile(TARGET, doraise=True)

check = open(TARGET).read()
if check.count("Normalize b first so the a count below matches.") != 1:
    sys.exit("vllm_lora_merged_set_lora: post-check failed")

print(f"vllm_lora_merged_set_lora: patched {TARGET} (PR #51423, 15 lines)")
