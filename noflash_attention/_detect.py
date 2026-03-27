"""
Structural analysis for FFN module detection.

Pre-filters candidate modules before runtime verification. This is an
optimization layer -- it eliminates obviously-wrong candidates (attention,
MoE, tiny projectors) so we dont waste time verifying them at runtime.

The correctness guarantee comes from runtime verification in ffn.py,
NOT from these heuristics. A false positive here is caught by verification.
A false negative here just means we skip a potential optimization.

All functions are pure (no side effects, no GPU calls) and CPU-testable.
"""

import torch.nn as nn
from typing import List, Optional, Tuple


def is_chunkable_candidate(module: nn.Module) -> Tuple[bool, str]:
    """
    Structural pre-filter: is this module worth runtime-verifying for FFN chunking?

    Returns (is_candidate, reason) where reason explains rejection.
    This is deliberately conservative -- we prefer false negatives (miss an
    optimization) over false positives (waste time verifying non-FFNs).
    """
    # 1. Reject MoE modules (cross-token expert routing)
    moe, reason = is_moe_module(module)
    if moe:
        return False, "moe: " + reason

    # 2. Reject modules with internal normalization (stats depend on full batch)
    if has_internal_normalization(module):
        return False, "has internal normalization (LayerNorm/RMSNorm/BatchNorm)"

    # 3. Reject attention modules (cross-token by definition)
    if is_attention_module(module):
        return False, "attention module (has Q/K/V projections)"

    # 4. Must have Linear layers that form a feedforward pattern
    linears = find_linear_layers(module)
    if len(linears) < 2:
        return False, "insufficient linear layers (%d, need >= 2)" % len(linears)

    # 5. Must have meaningful expansion (filters tiny projectors/embedders)
    ratio = expansion_ratio(linears)
    if ratio < 1.5:
        return False, "expansion ratio too low (%.1fx, need >= 1.5x)" % ratio

    # 6. Must not be a single Linear wrapped in a container
    #    (some models wrap Linear in Sequential for API consistency)
    direct_children = list(module.children())
    if len(direct_children) == 1 and isinstance(direct_children[0], nn.Linear):
        return False, "single Linear wrapper (not an FFN)"

    return True, "candidate"


def is_moe_module(module: nn.Module) -> Tuple[bool, str]:
    """
    Detect Mixture of Experts by structural signatures.

    MoE modules have a gating mechanism that selects experts based on
    cross-token statistics (top-k routing). Chunking tokens before the
    router changes expert assignments, producing WRONG results.

    Detection uses structural analysis, not name matching:
    - Signature 1: ModuleList of >= 2 similar sub-modules (experts) + a
      sibling module whose output dimension matches the expert count (gate)
    - Signature 2: Module has attributes commonly used for expert routing
      (top_k, num_experts, num_activated_experts, capacity_factor)
    """
    children = dict(module.named_children())

    # Find ModuleLists that could be expert lists
    expert_lists = []
    for name, child in children.items():
        if isinstance(child, nn.ModuleList) and len(child) >= 2:
            # Check if elements are structurally similar (same class)
            classes = set(type(e).__name__ for e in child)
            if len(classes) <= 2:  # allow 1-2 distinct classes
                expert_lists.append((name, child))

    if not expert_lists:
        # No expert lists found -- check for attribute-based signals
        if _has_routing_attributes(module):
            return True, "has routing attributes (top_k/num_experts)"
        return False, ""

    # Found potential expert list -- look for a gate
    for expert_name, experts in expert_lists:
        num_experts = len(experts)
        for gate_name, gate in children.items():
            if gate_name == expert_name:
                continue
            # Gate signature: has a weight with out_features == num_experts
            gate_out = _get_output_dim(gate)
            if gate_out is not None and gate_out == num_experts:
                return True, "experts=%s(%d), gate=%s" % (expert_name, num_experts, gate_name)

    # Expert list found but no matching gate -- still suspicious if routing attrs exist
    if _has_routing_attributes(module):
        return True, "expert list + routing attributes"

    return False, ""


def _has_routing_attributes(module: nn.Module) -> bool:
    """Check for attributes that indicate expert routing logic."""
    routing_attrs = {
        'top_k', 'num_experts', 'num_activated_experts', 'capacity_factor',
        'expert_capacity', 'num_selected_experts', 'router', 'gate',
        'moe_gate', 'routing_weights',
    }
    for attr in routing_attrs:
        if hasattr(module, attr):
            return True
    # Also check any child named 'gate' or 'router'
    for name, _ in module.named_children():
        if name in ('gate', 'router', 'moe_gate', 'routing'):
            return True
    return False


def _get_output_dim(module: nn.Module) -> Optional[int]:
    """Try to determine a module output dimension from its parameters."""
    # Direct Linear
    if isinstance(module, nn.Linear):
        return module.out_features
    # Module containing a Linear
    for child in module.children():
        if isinstance(child, nn.Linear):
            return child.out_features
    # Check weight attribute directly
    if hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
        try:
            return module.weight.shape[0]
        except (IndexError, AttributeError):
            pass
    return None


def has_internal_normalization(module: nn.Module) -> bool:
    """
    Check if module contains normalization layers as sub-modules.

    Normalization layers compute statistics across the batch/sequence
    dimension. Chunking would change these statistics, producing
    different results. Must reject.

    Checks both standard nn types and common custom implementations
    (RMSNorm variants used in LLaMA, Gemma, etc.)
    """
    builtin_norm_types = (
        nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    )
    for child in module.modules():
        if child is module:
            continue
        # Standard PyTorch norms
        if isinstance(child, builtin_norm_types):
            return True
        # Custom norms (RMSNorm, etc.) -- check class name as fallback.
        # This is the ONE place we use name-based detection, because custom
        # norm implementations dont share a base class. But its a conservative
        # check (false positives just skip a module, never wrong results).
        cls_name = type(child).__name__
        if any(tag in cls_name.lower() for tag in ('layernorm', 'rmsnorm', 'groupnorm', 'batchnorm')):
            return True
    return False


def is_attention_module(module: nn.Module) -> bool:
    """
    Reject attention modules. Attention is inherently cross-token
    (each token attends to all others). Our SDPA patch handles attention --
    FFN chunking must not wrap attention modules.

    Detection: module has >= 2 of the standard Q/K/V projection names.
    """
    child_names = set(name for name, _ in module.named_children())
    param_names = set(name.split('.')[0] for name, _ in module.named_parameters())
    all_names = child_names | param_names

    qkv_indicators = {
        'to_q', 'to_k', 'to_v', 'to_out',
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'q', 'k', 'v',  # Wan style
        'qkv', 'qkv_proj',
        'w_q', 'w_k', 'w_v',
    }
    matches = qkv_indicators & all_names
    return len(matches) >= 2


def find_linear_layers(module: nn.Module) -> List[nn.Linear]:
    """
    Find all Linear layers that are direct structural components of this module.

    Only searches immediate children and one level deep into Sequential/ModuleList
    containers. Does NOT recurse into arbitrary sub-modules (which could be
    entire sub-networks, not FFN components).
    """
    linears = []
    for child in module.children():
        if isinstance(child, nn.Linear):
            linears.append(child)
        elif isinstance(child, (nn.Sequential, nn.ModuleList)):
            for subchild in child:
                if isinstance(subchild, nn.Linear):
                    linears.append(subchild)
    return linears


def expansion_ratio(linears: List[nn.Linear]) -> float:
    """
    Compute the expansion ratio of an FFN from its Linear layers.

    Real FFNs expand then contract: e.g., 4096 -> 16384 -> 4096 (4x expansion).
    Small projectors have ratios near 1x. We require >= 1.5x to filter noise.
    """
    if not linears:
        return 0.0
    dims = []
    for linear in linears:
        if hasattr(linear, 'in_features') and hasattr(linear, 'out_features'):
            dims.extend([linear.in_features, linear.out_features])
    if not dims:
        return 0.0
    max_dim = max(dims)
    # Min of first Linear input and last Linear output (the "through" dimensions)
    first_in = linears[0].in_features if hasattr(linears[0], 'in_features') else max_dim
    last_out = linears[-1].out_features if hasattr(linears[-1], 'out_features') else max_dim
    min_io = min(first_in, last_out)
    if min_io == 0:
        return 0.0
    return max_dim / min_io
