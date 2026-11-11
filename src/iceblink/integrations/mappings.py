"""Architecture knowledge belongs here, not in monitor policies.

Candidate Qwen3.5-MoE mapping: vLLM's decoder returns the MLP branch and
the fused residual separately. Their sum is the residual after the block,
for both full attention and DeltaNet blocks. GPU reference validation is
required before recording this mapping as supported for a checkpoint.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMapping:
    identity: str
    architecture: str
    layer_path: str
    reconstruction: str
    tool_parser: str = "qwen3_coder"

    def layers(self, model):
        node = model
        for component in self.layer_path.split("."):
            node = getattr(node, component)
        return node


MAPPINGS = {
    "qwen3_5_moe.vllm.residual.v1": ModelMapping(
        "qwen3_5_moe.vllm.residual.v1", "Qwen3_5MoeForConditionalGeneration",
        "language_model.model.layers", "branch_plus_residual"),
    "qwen3_5_moe_text.vllm.residual.v1": ModelMapping(
        "qwen3_5_moe_text.vllm.residual.v1", "Qwen3_5MoeForCausalLM",
        "model.layers", "branch_plus_residual"),
    "qwen3.fixture.vllm.residual.v1": ModelMapping(
        "qwen3.fixture.vllm.residual.v1", "Qwen3ForCausalLM",
        "model.layers", "branch_plus_residual", "hermes"),
}


def residual_after_block(pair):
    if not isinstance(pair, tuple) or len(pair) != 2 or pair[1] is None:
        raise ValueError("Unsupported residual representation")
    return pair[0] + pair[1]
