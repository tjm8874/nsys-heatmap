#!/usr/bin/env python3
"""
cuda_hierarchy_viz.py — 7-Layer CUDA Hardware Hierarchy Efficiency Analyzer,
Bottleneck Inference Engine & GitHub Issue/PR Architecture Heatmap Generator.

Generates:
  1. Quantitative 7-layer efficiency JSON + prioritized bottleneck diagnosis
     (Host/PCIe -> Global VRAM Coalescing/Alignment -> L2 -> L1/SMEM Bank-Free ->
      Registers/Occupancy/Launch -> Warp SIMT Branchless -> Tensor Core/ALU/SFU).
  2. Crisp GitHub Issue/PR-ready SVG (and PNG when macOS `qlmanage` or Linux
     `rsvg-convert`/`resvg`/`inkscape` is available) in two modes:
       - `prefill-vs-decode`: Side-by-side Prefill vs Decode resource/BW scorecard
       - `commit-focus`: Architecture heatmap spotlighting exact blocks targeted by
         a commit/PR with Before -> After metrics and E2E speedup summary.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nsys_heatmap import audit as cuda_audit


# ---------------------------------------------------------------------------
# Static SASS Extractor for SIMT Divergence, Coalescing Width & Tensor Core Ops
# ---------------------------------------------------------------------------
SASS_LINE_RE = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(?:(@!?U?P(?:T|\d+))\s+)?([A-Z0-9_.]+)(.*)")

MMA_DECODE = [
    ("OMMA",    "NVFP4 (E2M1+UE4M3, m16n8k64)"),
    ("UTCQMMA", "Blackwell tcgen05 FP8/MXFP8 (TMEM)"),
    ("UTCOMMA", "Blackwell tcgen05 NVFP4 (TMEM)"),
    ("UTCHMMA", "Blackwell tcgen05 FP16/BF16 (TMEM)"),
    ("QGMMA",   "Hopper WGMMA FP8"),
    ("HGMMA",   "Hopper WGMMA FP16/BF16"),
    ("IGMMA",   "Hopper WGMMA INT8"),
    ("QMMA",    "MXFP8 / FP8 (E4M3+UE8M0, m16n8k32)"),
    ("IMMA",    "INT8 Tensor Core (s8->s32, m16n8k32)"),
    ("HMMA",    "FP16/BF16 Tensor Core (m16n8k16)"),
    ("IDP",     "INT8 dp4a SIMT Dot (MMVQ)"),
]


def extract_sass_hierarchy_metrics(sass_text: str) -> dict[str, Any]:
    """Extract static control-flow, global load width, SMEM, and TC metrics from SASS."""
    total_inst = 0
    divergent_bra = 0
    uniform_bra = 0
    bsync_ops = 0
    predicated_non_bra = 0
    mask_select_ops = 0
    call_ops = 0
    ldg_128_plus = 0
    ldg_64 = 0
    ldg_32 = 0
    ldg_8_16 = 0
    tma_ops = 0
    smem_vec_ops = 0
    smem_scalar_ops = 0
    swizzle_xor_ops = 0
    mma_counts: Counter[str] = Counter()

    for line in sass_text.splitlines():
        m = SASS_LINE_RE.search(line)
        if not m:
            continue
        pred, op, rest = m.group(1) or "", m.group(2), m.group(3) or ""
        total_inst += 1

        # Control flow: divergent warp branch vs uniform branch vs branchless mask/select
        if op.startswith("BRA"):
            if ".U" in op or pred.startswith(("@UP", "@!UP")) or pred in ("", "@PT"):
                uniform_bra += 1
            else:
                divergent_bra += 1
        elif op.startswith(("BSYNC", "WARPSYNC")):
            bsync_ops += 1
        else:
            if pred and not pred.startswith(("@PT", "@UP", "@!UP")):
                predicated_non_bra += 1
            if op.startswith(("SEL", "FSEL", "FMNMX", "IMNMX", "LOP3", "PLOP3", "SHFL", "VOTE", "REDUX")):
                mask_select_ops += 1

        if op.startswith("CALL"):
            call_ops += 1

        # Global memory load width proxy
        if "UTMALDG" in op or "UTMASTG" in op:
            tma_ops += 1
            ldg_128_plus += 1
        elif op.startswith(("LDG.", "LDGSTS.")):
            if re.search(r"\.(?:128|256)(?:\.|$)", op):
                ldg_128_plus += 1
            elif re.search(r"\.64(?:\.|$)", op):
                ldg_64 += 1
            elif re.search(r"\.(?:[US]?8|[US]?16)(?:\.|$)", op):
                ldg_8_16 += 1
            else:
                ldg_32 += 1

        # Shared memory vectorization & XOR swizzle clues
        if op.startswith(("LDSM", "STSM")) or (op.startswith(("LDS.", "STS.")) and (".128" in op or ".64" in op)):
            smem_vec_ops += 1
        elif op.startswith(("LDS", "STS")):
            smem_scalar_ops += 1
        if op.startswith("LOP3") and "0x3c" in rest.lower() or op.startswith("LOP3"):
            swizzle_xor_ops += 1

        for prefix, label in MMA_DECODE:
            if op.startswith(prefix):
                mma_counts[label] += 1
                break

    branchless_work = predicated_non_bra + mask_select_ops
    branchless_ratio = (
        100.0 * branchless_work / (branchless_work + divergent_bra * 4)
        if (branchless_work + divergent_bra) > 0 else 100.0
    )

    tot_ldg = ldg_128_plus + ldg_64 + ldg_32 + ldg_8_16
    if tot_ldg > 0:
        width_score = 100.0 * (1.0 * ldg_128_plus + 0.85 * ldg_64 + 0.65 * ldg_32 + 0.25 * ldg_8_16) / tot_ldg
    else:
        width_score = 100.0

    dominant_tc = mma_counts.most_common(1)[0][0] if mma_counts else "None (SIMT FP32/INT)"

    return {
        "total_instructions": total_inst,
        "divergent_bra": divergent_bra,
        "uniform_bra": uniform_bra,
        "bsync_ops": bsync_ops,
        "predicated_non_bra": predicated_non_bra,
        "mask_select_ops": mask_select_ops,
        "branchless_simt_score_pct": round(branchless_ratio, 1),
        "ldg_width_score_pct": round(width_score, 1),
        "ldg_counts": {"128b_or_tma": ldg_128_plus, "64b": ldg_64, "32b": ldg_32, "8_16b": ldg_8_16},
        "smem_vec_ops": smem_vec_ops,
        "smem_scalar_ops": smem_scalar_ops,
        "call_ops": call_ops,
        "dominant_compute_unit": dominant_tc,
        "mma_breakdown": dict(mma_counts),
    }


# ---------------------------------------------------------------------------
# 7-Layer Bottleneck Inference Engine
# ---------------------------------------------------------------------------
LAYER_ORDER = [
    ("host_pcie",     "1. Host RAM & PCIe Bus",        "Pinned DMA / HtoD-DtoH Overlap"),
    ("global_vram",   "2. Device Global VRAM",         "DRAM BW / Coalescing / Alignment"),
    ("l2_cache",      "3. L2 Cache & Tile Locality",   "Hit Rate / 2D & LPT Swizzle"),
    ("smem_l1",       "4. L1 Cache & Shared Memory",   "Discrete Carveout / Bank-Free %"),
    ("regs_occ",      "5. Register File & Occupancy",  "Regs/Thread / Spills / Launch Eff"),
    ("warp_simt",     "6. Warp SIMT & Schedulers",     "Active Mask / Branchless vs if-else"),
    ("compute_cores", "7. Tensor Cores, ALU & SFU",    "Dtype & M×N×K Tile / GEMM Eff %"),
]


def infer_bottlenecks(phase_name: str, layers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank bottlenecks by combining low efficiency score with phase time weight."""
    candidates: list[dict[str, Any]] = []
    for key, title, _sub in LAYER_ORDER:
        node = layers.get(key)
        if not node or node.get("inactive", False) or node.get("expected_regime_bound", False):
            continue
        eff = float(node.get("efficiency_pct", 100.0))
        weight = float(node.get("time_weight_pct", 15.0))
        headroom = max(0.0, 88.0 - eff)
        impact_score = round((weight / 100.0) * headroom, 2)
        if eff < 75.0 and impact_score >= 1.5:
            candidates.append({
                "phase": phase_name,
                "layer_key": key,
                "layer_title": title,
                "efficiency_pct": eff,
                "time_weight_pct": weight,
                "estimated_gain_potential_pct": impact_score,
                "diagnosis": node.get("diagnosis") or f"Low efficiency ({eff:.0f}%) on {weight:.0f}% time-share stage.",
                "action": node.get("action") or "Profile with ncu / sass-audit and apply targeted lever.",
            })
    candidates.sort(key=lambda x: x["estimated_gain_potential_pct"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Built-In Empirical Basalt Presets (Prefill vs Decode & Wave 3 Commit Focus)
# ---------------------------------------------------------------------------
def get_preset_profile(name: str) -> dict[str, Any]:
    if name in ("prefill-vs-decode", "basalt-prefill-vs-decode"):
        return {
            "mode": "prefill-vs-decode",
            "title": "Basalt (Inkling 2 Mini) — CUDA Hierarchy Efficiency: Prefill vs Decode",
            "subtitle": "NVIDIA RTX PRO 5000 72GB GDDR7 (sm_120, 110 SMs, 300W) | CUDA 13.4 | Identifying Regime Bottlenecks",
            "footer": "Green (>=80%): Near Hardware Ceiling  •  Amber (55–79%): Tunable Headroom  •  Red (<55%): Primary Bottleneck  •  Slate: Inactive",
            "focus_targets": ["global_vram", "regs_occ", "smem_l1", "warp_simt"],
            "focus_banner": "Regime Contrast: Prefill is Tensor-Core & SMEM-Pipeline Bound • Decode is Weight-VRAM, Launch-Tail & KV-Coalescing Bound",
            "left_label": "PREFILL (pp16384, Chunk T=4096)",
            "right_label": "DECODE (--spec 1 / --spec 8, T=1..8)",
            "left_summary": "14,121 tok/s • GEMM/Attn Share: 83.2% • Compute-Bound",
            "right_summary": "215 / 493 tok/s • Glue Share: 25–32% • BW & Launch-Bound",
            "left": {
                "host_pcie": {
                    "inactive": True, "efficiency_pct": 100.0, "time_weight_pct": 0.0,
                    "headline": "VRAM-Resident (0 B PCIe)",
                    "detail": "17.0 GiB weights resident (Spilled 5070Ti: 53.7% HtoD hidden)",
                    "tag": "[nsys]",
                },
                "global_vram": {
                    "efficiency_pct": 91.0, "time_weight_pct": 18.0,
                    "headline": "Coalesced: 96% • Align: 128B SoA",
                    "detail": "TMA 2D + 4-lane 16B cp.async.cg • 1,340+ GB/s on norms/mix",
                    "tag": "[SASS/nsys]",
                },
                "l2_cache": {
                    "efficiency_pct": 88.0, "time_weight_pct": 15.0,
                    "headline": "2D Swizzle (G=8) + LPT KV Order",
                    "detail": "ctrl_down 589KB 100% L2-hit; 32K FlashAttn LPT +26.1%",
                    "tag": "[ABBA]",
                },
                "smem_l1": {
                    "efficiency_pct": 90.0, "time_weight_pct": 30.0,
                    "headline": "Bank-Free: ~100% • 128B/64B XOR",
                    "detail": "100 KiB carveout (28 KiB L1 safe via TMA/.cg) • 4-stage far KV",
                    "tag": "[SASS]",
                },
                "regs_occ": {
                    "efficiency_pct": 89.0, "time_weight_pct": 25.0,
                    "headline": "128–252 Regs • 0 B Spill • 1–2 CTA/SM",
                    "detail": "q8_gemm: 512 thr × 128 regs = 100% RF • Launches: 37.4k -> 9.8k",
                    "tag": "[res-usage]",
                },
                "warp_simt": {
                    "efficiency_pct": 93.0, "time_weight_pct": 15.0,
                    "headline": "SIMT Eff: ~95% • Tile-Hoisted Mask",
                    "detail": "0 inner-loop int64%cap • Branchless whole-tile causal softmax",
                    "tag": "[SASS]",
                },
                "compute_cores": {
                    "efficiency_pct": 84.0, "time_weight_pct": 78.0,
                    "headline": "NVFP4 m16n8k64 • MXFP8/INT8 m16n8k32",
                    "detail": "Tiles 128×256 & 128×128 • 249–404 TOP/s • 0 libm CALLs (MUFU)",
                    "tag": "[Microbench]",
                },
            },
            "right": {
                "host_pcie": {
                    "inactive": True, "efficiency_pct": 100.0, "time_weight_pct": 0.0,
                    "headline": "VRAM-Resident (0 B PCIe)",
                    "detail": "Spilled mode: 9KB DtoH overlapped w/ router + float4 SYS merge",
                    "tag": "[nsys]",
                },
                "global_vram": {
                    "efficiency_pct": 54.0, "time_weight_pct": 48.0,
                    "headline": "q8_planes: 85% BW • NVFP4 AoS: 41% BW",
                    "detail": "36B block_nvfp4 forces LDG.32 (549 GB/s); q8_planes hits 1.16 TB/s",
                    "tag": "[Bottleneck]",
                    "diagnosis": "36B AoS block_nvfp4 breaks 16B alignment in native_down_sub8<40> (41% DRAM BW).",
                    "action": "Repack NVFP4 into 128B-aligned planar SoA to unlock LDG.128 (uint4).",
                },
                "l2_cache": {
                    "efficiency_pct": 82.0, "time_weight_pct": 12.0,
                    "headline": "Inter-Kernel Act Warm in L2",
                    "detail": "9–36 KB token activations stay L2-resident across layer visits",
                    "tag": "[Arch]",
                },
                "smem_l1": {
                    "efficiency_pct": 78.0, "time_weight_pct": 10.0,
                    "headline": "Low SMEM (0–49 KiB) • L1=64–128 KiB",
                    "detail": "attn.cu cp.async V cached in L1; MMVQ uses warp shuffles",
                    "tag": "[SASS]",
                },
                "regs_occ": {
                    "efficiency_pct": 46.0, "time_weight_pct": 32.0,
                    "headline": "0 B Spill • Glue Storm: ~28% Time",
                    "detail": "Multiple 3–6 us micro-kernels/visit create grid-drain bubbles",
                    "tag": "[Bottleneck]",
                    "diagnosis": "9 micro-kernels per visit (~3–6 us each) spend ~25–32% of Decode in launch/tail drain.",
                    "action": "Mega-fuse (ctrl_down+up+blend) & (post_attn+mix_norm) + expand PDL preload.",
                },
                "warp_simt": {
                    "efficiency_pct": 64.0, "time_weight_pct": 24.0,
                    "headline": "432 PRMT LUT Ops in NVFP4 Split<40>",
                    "detail": "Branchless dp4a in Q8, but FP4 nibble LUT burns SIMT ALU cycles",
                    "tag": "[SASS]",
                    "diagnosis": "432 PRMT byte-permutes in llama.cpp-style FP4 LUT dequantization.",
                    "action": "Replace PRMT LUT with native sm_120a `cvt.rn.f16x2.e2m1x2` (`F2FP`).",
                },
                "compute_cores": {
                    "efficiency_pct": 38.0, "time_weight_pct": 20.0,
                    "expected_regime_bound": True,
                    "headline": "T=1: SIMT dp4a/FFMA • T>=4: m16n8k32",
                    "detail": "M=1..3 underfills m16 Tensor Core rows; gated by weight DRAM BW",
                    "tag": "[Expected @ T=1]",
                },
            },
        }

    if name in ("commit-focus", "basalt-wave3-pr"):
        return {
            "mode": "commit-focus",
            "title": "PR Commit Focus Heatmap — Basalt Wave 3 (`e5901909 -> 1f818f7a`)",
            "subtitle": "7 Bitwise-Exact Commits on Blackwell (sm_120) • RTX PRO 5000 (+4.9% @ 64K) & Spilled RTX 5070 Ti (+34.6% @ 16K)",
            "footer": "Highlighted Cyan/Gold nodes indicate exact GPU hierarchy bottlenecks targeted by this PR • 100% Bitwise Parity Verified",
            "focus_targets": ["host_pcie", "global_vram", "smem_l1", "warp_simt", "compute_cores"],
            "focus_banner": "PR TARGETS: [1] Async PCIe Overlap  [2] SMEM Row-Sweep Locality  [4] 4-Stage KV + Bank-Free Router  [6] Ring Modulo Hoist  [7] 16-Warp Q8 TMA",
            "left_label": "BEFORE (PR #579 Baseline)",
            "right_label": "AFTER (Wave 3 `1f818f7a`)",
            "left_summary": "PRO 5000 32K: 12,568 t/s • 5070Ti Spilled 16K: 5,286 t/s",
            "right_summary": "PRO 5000 32K: 12,987 t/s (+3.3%) • 5070Ti 16K: 7,114 t/s (+34.6%)",
            "left": {
                "host_pcie": {
                    "efficiency_pct": 18.0, "time_weight_pct": 45.8,
                    "headline": "0.0% HtoD Overlap (1,357 ms Stall)",
                    "detail": "Synchronous layer staging + all-or-nothing PCIe copy on partial promo",
                    "tag": "[Spilled 5070Ti]",
                    "diagnosis": "1,357 ms synchronous PCIe HtoD stall (0% overlap) + redundant RAM copies",
                    "action": "h2d_stream_ + exp_done/h2d_done lookahead + DtoD for promoted slots (+34.6% E2E)",
                },
                "global_vram": {
                    "efficiency_pct": 48.0, "time_weight_pct": 8.0,
                    "headline": "ctrl_up_blend: 8-Row Stride (646 GB/s)",
                    "detail": "Each warp strided 8 rows × 9.2 KB apart; scalar moe_cpu_merge (2.67 us)",
                    "tag": "[Row-Stride]",
                    "diagnosis": "8-row strided warp accesses in ctrl_up_blend + scalar PCIe merge",
                    "action": "Stage gate/forget in 35.8 KiB SMEM for 1-row 256B sweep (2.31x) + float4 PDL",
                },
                "l2_cache": {
                    "efficiency_pct": 74.0, "time_weight_pct": 15.0,
                    "headline": "Sequential Head Order in FlashAttn",
                    "detail": "Long-history causal CTAs scheduled late -> 32K tail wave stragglers",
                    "tag": "[Tail Wave]",
                },
                "smem_l1": {
                    "efficiency_pct": 52.0, "time_weight_pct": 18.0,
                    "headline": "2-Stage Far KV • Sync Router (288 Bars)",
                    "detail": "2 barriers/tile in attn_prompt; naive FD_K=16 outer loop in router",
                    "tag": "[Pipeline Stall]",
                    "diagnosis": "2-barrier KV loop & 288-barrier synchronous router GEMM stall warps",
                    "action": "1-barrier 4-stage cp.async KV ring + 3-stage bank-free LDS.128 router (5.56x)",
                },
                "regs_occ": {
                    "efficiency_pct": 82.0, "time_weight_pct": 14.0,
                    "headline": "2-CTA 64×64 Q8 TMA (168 Regs, 0 Spill)",
                    "detail": "Good for narrow shapes, but reloads activations 2x on [8664×2304]",
                    "tag": "[Tile Width]",
                },
                "warp_simt": {
                    "efficiency_pct": 58.0, "time_weight_pct": 15.0,
                    "headline": "Inner-Loop int64 % cap + Per-Key Mask",
                    "detail": "64-bit modulo & fresh/ok branches inside every 16B cp.async piece",
                    "tag": "[ALU/Branch]",
                    "diagnosis": "Per-piece int64 % cap division & mask branches inside FlashAttn inner loop",
                    "action": "Hoist uint32 slot wrap to tile header + branchless whole-tile far softmax (+26.1%)",
                },
                "compute_cores": {
                    "efficiency_pct": 79.0, "time_weight_pct": 30.0,
                    "headline": "Q8 TMA: 232 TOP/s • Router: 4.4 TFLOP/s",
                    "detail": "Attention 32K: 130.5 TFLOP/s; Q8 quantizer scalar warp loop (40.1 us)",
                    "tag": "[TC/SIMT]",
                },
            },
            "right": {
                "host_pcie": {
                    "efficiency_pct": 86.0, "time_weight_pct": 27.8,
                    "headline": "53.7% HtoD Hidden (-735 ms Stall!)",
                    "detail": "0-VRAM h2d_stream_ lookahead + DtoD (447 GB/s) + coalesced HtoD",
                    "tag": "[+34.6% E2E]",
                },
                "global_vram": {
                    "efficiency_pct": 94.0, "time_weight_pct": 3.5,
                    "headline": "1-Row 256B Sweep (1,493 GB/s, 2.31x!)",
                    "detail": "SMEM-staged ctrl_up_blend (73.2 us) + float4/PDL merge (1.25 us, 2.14x)",
                    "tag": "[2.31x Faster]",
                },
                "l2_cache": {
                    "efficiency_pct": 90.0, "time_weight_pct": 12.0,
                    "headline": "L2-Aware LPT CTA Scheduling",
                    "detail": "Longest-history query tiles launched first across KV heads",
                    "tag": "[Balanced]",
                },
                "smem_l1": {
                    "efficiency_pct": 91.0, "time_weight_pct": 13.0,
                    "headline": "1-Barrier 4-Stage KV + 3-Stage Router",
                    "detail": "68 KiB union Smem 4-stage cp.async; bank-free LDS.128 router (5.56x!)",
                    "tag": "[5.56x Router]",
                },
                "regs_occ": {
                    "efficiency_pct": 93.0, "time_weight_pct": 13.0,
                    "headline": "16-Warp 128×128 TMA (128 Regs = 100% RF)",
                    "detail": "4 warps/sub-partition on sm_120 + PDL NPRE=1 weight preload on MMVQ",
                    "tag": "[0 B Spill]",
                },
                "warp_simt": {
                    "efficiency_pct": 94.0, "time_weight_pct": 12.0,
                    "headline": "Tile-Hoisted uint32 Slot + Branchless Far",
                    "detail": "Zero inner-loop modulo; branchless whole-tile softmax on far prefix",
                    "tag": "[+26.1% @ 32K]",
                },
                "compute_cores": {
                    "efficiency_pct": 91.0, "time_weight_pct": 28.0,
                    "headline": "Q8 TMA: 249.2 TOP/s • Attn: 164.5 TFLOP/s",
                    "detail": "Router: 97.2 us (24.5 TFLOP/s FP32) • vec4 Q8 quantizer: 11.8 us (3.4x)",
                    "tag": "[Bitwise Exact]",
                },
            },
        }

    raise ValueError(f"Unknown preset: {name}")


# ---------------------------------------------------------------------------
# Zero-Dependency GitHub-Ready SVG + Optional PNG Renderer
# ---------------------------------------------------------------------------
def eff_palette(pct: float, inactive: bool = False) -> tuple[str, str, str]:
    """Return (fill_bg, bar_color, text_badge_color) for an efficiency percentage."""
    if inactive:
        return ("#162032", "#475569", "#94a3b8")
    if pct >= 80.0:
        return ("#0c2924", "#10b981", "#34d399")  # Emerald: Optimal / Saturated
    if pct >= 55.0:
        return ("#2b2110", "#f59e0b", "#fbbf24")  # Amber: Tunable Headroom
    return ("#31131b", "#ef4444", "#f87171")      # Crimson: Active Bottleneck


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def render_hierarchy_svg(profile: dict[str, Any]) -> str:
    width = 1520
    row_h = 96
    row_gap = 12
    top_y = 166
    n_rows = len(LAYER_ORDER)

    is_commit_mode = profile.get("mode") == "commit-focus"
    left_bots = infer_bottlenecks("Baseline" if is_commit_mode else "Prefill", profile["left"])
    right_bots = [] if is_commit_mode else infer_bottlenecks("Decode", profile["right"])
    all_bots = sorted(left_bots + right_bots, key=lambda b: b["estimated_gain_potential_pct"], reverse=True)[:3]

    bot_panel_h = 152 if all_bots else 56
    height = top_y + n_rows * (row_h + row_gap) + bot_panel_h + 46

    focus_set = set(profile.get("focus_targets", []))

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif">',
        "<defs>",
        '  <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">',
        '    <stop offset="0%" stop-color="#090d16"/>',
        '    <stop offset="100%" stop-color="#0f172a"/>',
        "  </linearGradient>",
        '  <filter id="focusGlow" x="-4%" y="-10%" width="108%" height="120%">',
        '    <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#38bdf8" flood-opacity="0.45"/>',
        "  </filter>",
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="16" fill="url(#bgGrad)" stroke="#1e293b" stroke-width="2"/>',
        # Header
        f'<text x="36" y="44" fill="#f8fafc" font-size="23" font-weight="700">{esc(profile["title"])}</text>',
        f'<text x="36" y="70" fill="#94a3b8" font-size="14">{esc(profile.get("subtitle", ""))}</text>',
    ]

    # Focus banner pill
    banner = profile.get("focus_banner", "")
    if banner:
        b_stroke = "#f59e0b" if is_commit_mode else "#38bdf8"
        b_fill = "#271e0b" if is_commit_mode else "#0c2338"
        b_txt = "#fcd34d" if is_commit_mode else "#7dd3fc"
        parts.append(
            f'<rect x="36" y="84" width="{width - 72}" height="32" rx="8" fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.2"/>'
            f'<text x="52" y="105" fill="{b_txt}" font-size="13.5" font-weight="600">{esc(banner)}</text>'
        )

    # Column Headers (Hardware Spine | Left Phase | Right Phase)
    spine_x, spine_w = 36, 320
    left_x, col_w = 376, 544
    right_x = 940

    parts.extend([
        f'<text x="{spine_x + 8}" y="148" fill="#cbd5e1" font-size="14" font-weight="700">CUDA HARDWARE HIERARCHY</text>',
        f'<text x="{left_x + 8}" y="140" fill="#e2e8f0" font-size="15" font-weight="700">{esc(profile["left_label"])}</text>',
        f'<text x="{left_x + 8}" y="157" fill="#94a3b8" font-size="12.5">{esc(profile.get("left_summary", ""))}</text>',
        f'<text x="{right_x + 8}" y="140" fill="#e2e8f0" font-size="15" font-weight="700">{esc(profile["right_label"])}</text>',
        f'<text x="{right_x + 8}" y="157" fill="#94a3b8" font-size="12.5">{esc(profile.get("right_summary", ""))}</text>',
    ])

    # Render each of the 7 Hardware Layers
    for idx, (lkey, ltitle, lsub) in enumerate(LAYER_ORDER):
        y = top_y + idx * (row_h + row_gap)
        focused = lkey in focus_set

        # Connector arrow between hardware spine blocks
        if idx > 0:
            ay = y - row_gap
            parts.append(
                f'<line x1="{spine_x + spine_w // 2}" y1="{ay}" x2="{spine_x + spine_w // 2}" y2="{y}" '
                'stroke="#334155" stroke-width="2" stroke-dasharray="3,3"/>'
            )

        # Spine card
        sp_stroke = "#38bdf8" if focused else "#334155"
        sp_sw = "2.2" if focused else "1.2"
        sp_fill = "#112236" if focused else "#131c2e"
        parts.append(
            f'<rect x="{spine_x}" y="{y}" width="{spine_w}" height="{row_h}" rx="10" '
            f'fill="{sp_fill}" stroke="{sp_stroke}" stroke-width="{sp_sw}"'
            + (' filter="url(#focusGlow)"' if focused else "") + "/>"
        )
        parts.append(f'<text x="{spine_x + 16}" y="{y + 34}" fill="#f1f5f9" font-size="15" font-weight="700">{esc(ltitle)}</text>')
        parts.append(f'<text x="{spine_x + 16}" y="{y + 56}" fill="#94a3b8" font-size="12.5">{esc(lsub)}</text>')
        if focused:
            badge_label = "TARGETED BY COMMIT" if is_commit_mode else "KEY REGIME DIFF"
            badge_col = "#f59e0b" if is_commit_mode else "#38bdf8"
            parts.append(
                f'<rect x="{spine_x + 16}" y="{y + 66}" width="156" height="20" rx="5" fill="#090d16" stroke="{badge_col}" stroke-width="1"/>'
                f'<text x="{spine_x + 25}" y="{y + 80}" fill="{badge_col}" font-size="10.5" font-weight="700">{badge_label}</text>'
            )

        # Render Left & Right Metric Cards
        l_node = profile["left"].get(lkey, {})
        r_node = profile["right"].get(lkey, {})
        for side_idx, (cx, node) in enumerate(((left_x, l_node), (right_x, r_node))):
            inactive = bool(node.get("inactive", False))
            eff = float(node.get("efficiency_pct", 0.0))
            bg_col, bar_col, txt_col = eff_palette(eff, inactive=inactive)

            c_stroke = ("#f59e0b" if (focused and is_commit_mode and side_idx == 1)
                        else ("#38bdf8" if focused else "#26334d"))
            c_sw = "2.2" if (focused and (not is_commit_mode or side_idx == 1)) else "1.1"

            parts.append(
                f'<rect x="{cx}" y="{y}" width="{col_w}" height="{row_h}" rx="10" '
                f'fill="{bg_col}" stroke="{c_stroke}" stroke-width="{c_sw}"/>'
            )

            # Efficiency meter bar
            bar_x, bar_y, bar_w, bar_h = cx + 16, y + 14, col_w - 150, 12
            fill_w = max(0.0, min(1.0, eff / 100.0)) * bar_w
            parts.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="6" fill="#090d16"/>')
            if not inactive and fill_w > 0:
                parts.append(f'<rect x="{bar_x}" y="{bar_y}" width="{fill_w:.1f}" height="{bar_h}" rx="6" fill="{bar_col}"/>')

            # Efficiency % readout + delta badge on Right card in commit-focus mode
            pct_str = "N/A" if inactive else f"{eff:.0f}% Eff"
            if is_commit_mode and side_idx == 1 and not inactive and not l_node.get("inactive", False):
                delta = eff - float(l_node.get("efficiency_pct", 0.0))
                if abs(delta) >= 1.0:
                    pct_str = f"{eff:.0f}% ({delta:+.0f}pt)"

            parts.append(
                f'<text x="{cx + col_w - 16}" y="{y + 25}" text-anchor="end" fill="{txt_col}" '
                f'font-size="14.5" font-weight="700" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{esc(pct_str)}</text>'
            )

            # Headline & Detail lines
            parts.append(
                f'<text x="{cx + 16}" y="{y + 52}" fill="#f8fafc" font-size="14" font-weight="700">{esc(node.get("headline", ""))}</text>'
            )
            tag = node.get("tag", "")
            if tag:
                parts.append(
                    f'<text x="{cx + col_w - 16}" y="{y + 52}" text-anchor="end" fill="#cbd5e1" font-size="11.5" '
                    f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{esc(tag)}</text>'
                )
            parts.append(
                f'<text x="{cx + 16}" y="{y + 76}" fill="#cbd5e1" font-size="12.2">{esc(node.get("detail", ""))}</text>'
            )

    # Bottom Inferred Bottleneck Action Queue Panel
    by = top_y + n_rows * (row_h + row_gap) + 8
    if all_bots:
        hdr_txt = (
            "BOTTLENECKS TARGETED & RESOLVED BY THIS PR (Ranked by Baseline Impact):"
            if is_commit_mode else
            "AUTOMATED BOTTLENECK INFERENCE (Ranked by Time-Share × Efficiency Headroom):"
        )
        hdr_col = "#fcd34d" if is_commit_mode else "#f87171"
        parts.append(
            f'<rect x="36" y="{by}" width="{width - 72}" height="{bot_panel_h - 12}" rx="10" '
            'fill="#131c2e" stroke="#334155" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="54" y="{by + 26}" fill="{hdr_col}" font-size="13.2" font-weight="700">{esc(hdr_txt)}</text>'
        )
        for bi, b in enumerate(all_bots):
            ly = by + 50 + bi * 28
            line = (
                f"#{bi + 1} [{b['phase']} | {b['layer_title']}] ({b['efficiency_pct']:.0f}% eff, "
                f"{b['time_weight_pct']:.0f}% wt): {b['diagnosis']} -> {b['action']}"
            )
            if len(line) > 218:
                line = line[:215] + "..."
            parts.append(
                f'<text x="54" y="{ly}" fill="#e2e8f0" font-size="11.5" '
                f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{esc(line)}</text>'
            )

    # Footer legend
    parts.append(
        f'<text x="36" y="{height - 14}" fill="#64748b" font-size="12">{esc(profile.get("footer", ""))}</text>'
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def try_rasterize_svg_to_png(svg_path: Path, png_path: Path, width_px: int = 1920) -> bool:
    """Convert SVG to full-aspect-ratio PNG using available system tools (zero Python deps)."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    # 1. rsvg-convert (Linux/macOS)
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-w", str(width_px), "-o", str(png_path), str(svg_path)], check=True)
        return png_path.is_file()
    # 2. resvg
    if shutil.which("resvg"):
        subprocess.run(["resvg", "-w", str(width_px), str(svg_path), str(png_path)], check=True)
        return png_path.is_file()
    # 3. macOS built-in `sips` (preserves exact aspect ratio without QuickLook square-cropping!)
    if shutil.which("sips"):
        subprocess.run(
            ["sips", "-s", "format", "png", str(svg_path), "--out", str(png_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if width_px and width_px != 1520 and png_path.is_file():
            subprocess.run(
                ["sips", "--resampleWidth", str(width_px), str(png_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        return png_path.is_file()
    # 4. Inkscape
    if shutil.which("inkscape"):
        subprocess.run(
            ["inkscape", str(svg_path), f"--export-width={width_px}", f"--export-filename={png_path}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return png_path.is_file()
    return False


# ---------------------------------------------------------------------------
# Auto-Profile Engine: nsys SQLite + SASS/res-usage + optional ncu CSV -> 7 Layers
# ---------------------------------------------------------------------------
GEMM_ATTN_KEYWORDS = ("gemm", "mma", "attn", "flash", "mmq", "mmvq", "wgemm", "cutlass", "cublas", "nvjet", "moe_fp4", "moe_mxfp8")
GEMV_DECODE_KEYWORDS = ("mmvq", "gemv", "small_q8", "vec_dot", "split", "sub8")


def parse_sass_and_res_by_fn(
    binary: str | None, sass_file: str | None, res_file: str | None
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    """Return (per_fn_sass_metrics, per_fn_res_usage, aggregate_sass_metrics)."""
    sass_text = ""
    res_text = ""
    if binary:
        res_text = subprocess.check_output(["cuobjdump", "-res-usage", binary], text=True, stderr=subprocess.STDOUT)
        sass_text = subprocess.check_output(["cuobjdump", "-sass", binary], text=True, stderr=subprocess.STDOUT)
    else:
        if sass_file:
            sass_text = Path(sass_file).read_text(encoding="utf-8", errors="replace")
        if res_file:
            res_text = Path(res_file).read_text(encoding="utf-8", errors="replace")

    res_map: dict[str, dict[str, int]] = {}
    cur_fn: str | None = None
    for line in res_text.splitlines():
        m_fn = re.search(r"Function\s+([^\s:]+):", line)
        if m_fn:
            cur_fn = m_fn.group(1)
            res_map.setdefault(cur_fn, {"regs": 0, "smem_static": 0, "stack": 0, "spill_stores": 0, "spill_loads": 0})
            continue
        if cur_fn:
            for key, pat in [
                ("regs", r"REG:(\d+)"),
                ("smem_static", r"SHARED:(\d+)|SMEM:(\d+)"),
                ("stack", r"STACK:(\d+)"),
                ("spill_stores", r"SPILL_STORES:(\d+)"),
                ("spill_loads", r"SPILL_LOADS:(\d+)"),
            ]:
                m = re.search(pat, line)
                if m:
                    val = next(g for g in m.groups() if g is not None)
                    res_map[cur_fn][key] = int(val)

    chunks: dict[str, list[str]] = defaultdict(list)
    active_fn: str | None = None
    for line in sass_text.splitlines():
        m_hdr = re.search(r"Function\s*:\s*([^\s]+)", line)
        if m_hdr:
            active_fn = m_hdr.group(1)
            continue
        if active_fn:
            chunks[active_fn].append(line)

    fn_sass = {fn: extract_sass_hierarchy_metrics("\n".join(lines)) for fn, lines in chunks.items()}
    agg_sass = extract_sass_hierarchy_metrics(sass_text) if sass_text else extract_sass_hierarchy_metrics("")
    return fn_sass, res_map, agg_sass


def parse_ncu_csv(csv_path: str | None) -> dict[str, float]:
    """Parse optional `ncu --csv` output (skipping `==PROF==` preamble lines)."""
    if not csv_path:
        return {}
    lines = Path(csv_path).read_text(encoding="utf-8", errors="replace").splitlines()
    hdr_idx = None
    for idx, ln in enumerate(lines):
        if "Metric Name" in ln and "Metric Value" in ln:
            hdr_idx = idx
            break
    if hdr_idx is None:
        return {}

    acc: dict[str, list[float]] = defaultdict(list)
    reader = csv.DictReader(lines[hdr_idx:])
    for row in reader:
        mname = (row.get("Metric Name") or "").strip()
        mval_raw = (row.get("Metric Value") or "").strip().replace(",", "").rstrip("%")
        if not mname or not mval_raw:
            continue
        try:
            acc[mname].append(float(mval_raw))
        except ValueError:
            continue
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def match_kernel_entry(kname: str, table: dict[str, Any]) -> Any | None:
    if not table:
        return None
    if kname in table:
        return table[kname]
    # Extract C++ identifier stem before '(' or '<'
    stem = kname.split("(")[0].split("<")[0].split("::")[-1].strip()
    if len(stem) >= 4:
        for fn_key, val in table.items():
            if stem in fn_key:
                return val
    return None


def build_phase_from_telemetry(
    sqlite_path: str,
    arch_input: str,
    gpu_preset_name: str,
    pcie_ceiling_gbs: float = 28.0,
    device_filter: int | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    binary: str | None = None,
    sass_file: str | None = None,
    res_file: str | None = None,
    ncu_csv: str | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Ingest nsys SQLite (+ optional SASS/res-usage + optional ncu CSV) into a 7-layer dict."""
    arch = cuda_audit.normalize_arch(arch_input)
    preset = cuda_audit.GPU_PRESETS.get(gpu_preset_name, cuda_audit.GPU_PRESETS["rtx_pro_5000_300w"])
    num_sms = int(preset.get("sms", 110))

    fn_sass, res_map, agg_sass = parse_sass_and_res_by_fn(binary, sass_file, res_file)
    has_sass = agg_sass["total_instructions"] > 0
    ncu = parse_ncu_csv(ncu_csv)

    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cur = conn.cursor()
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        conn.close()
        raise ValueError(f"CUPTI_ACTIVITY_KIND_KERNEL missing in {sqlite_path}")

    kcols = {r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    has_str = "StringIds" in tables
    name_col = "demangledName" if "demangledName" in kcols else "shortName"
    grid_expr = "COALESCE(k.gridX * k.gridY * k.gridZ, 0)" if {"gridX", "gridY", "gridZ"} <= kcols else "0"
    blk_expr = "COALESCE(k.blockX * k.blockY * k.blockZ, 256)" if {"blockX", "blockY", "blockZ"} <= kcols else "256"
    reg_expr = "COALESCE(k.registersPerThread, 64)" if "registersPerThread" in kcols else "64"
    smem_expr = (
        "COALESCE(k.dynamicSharedMemory, 0) + COALESCE(k.staticSharedMemory, 0)"
        if {"dynamicSharedMemory", "staticSharedMemory"} <= kcols else "0"
    )
    dev_expr = "COALESCE(k.deviceId, 0)" if "deviceId" in kcols else "0"

    if has_str:
        q = f"""
            SELECT COALESCE(s.value, CAST(k.{name_col} AS TEXT)), k.start, k.end, {dev_expr},
                   {grid_expr}, {blk_expr}, {reg_expr}, {smem_expr}
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON k.{name_col} = s.id
        """
    else:
        q = f"SELECT CAST(k.{name_col} AS TEXT), k.start, k.end, {dev_expr}, {grid_expr}, {blk_expr}, {reg_expr}, {smem_expr} FROM CUPTI_ACTIVITY_KIND_KERNEL k"

    # First pass: collect clipped kernel launches grouped by device
    launches_by_dev: dict[int, list[tuple[str, int, int, int, int, int, int]]] = defaultdict(list)
    ns_by_dev: Counter[int] = Counter()
    for kname, s_ns, e_ns, dev_id, grid_sz, blk_sz, regs, smem in cur.execute(q):
        clipped = cuda_audit.clip_interval(int(s_ns), int(e_ns), start_ns, end_ns)
        if clipped is None:
            continue
        dev = int(dev_id)
        if device_filter is not None and dev != device_filter:
            continue
        cs, ce = clipped
        launches_by_dev[dev].append((str(kname), cs, ce, int(grid_sz), int(blk_sz), int(regs), int(smem)))
        ns_by_dev[dev] += (ce - cs)

    if not launches_by_dev:
        conn.close()
        raise ValueError(f"No kernel launches found in {sqlite_path} for given window/device.")

    target_dev = device_filter if device_filter is not None else ns_by_dev.most_common(1)[0][0]
    k_events = launches_by_dev[target_dev]
    compute_merged = cuda_audit.merge_intervals([(cs, ce) for _, cs, ce, *_ in k_events])
    compute_active_ns = sum(e - s for s, e in compute_merged)
    total_k_ns = sum(ce - cs for _, cs, ce, *_ in k_events)

    # Memcpy intervals on target_dev
    mc_ivs: list[tuple[int, int, float]] = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        mcols = {r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
        mdev = "COALESCE(deviceId, 0)" if "deviceId" in mcols else str(target_dev)
        for ckind, nbytes, s_ns, e_ns, d_id in cur.execute(
            f"SELECT copyKind, bytes, start, end, {mdev} FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ):
            if int(d_id) != target_dev or int(ckind) not in (1, 2):  # HtoD / DtoH
                continue
            clipped = cuda_audit.clip_interval(int(s_ns), int(e_ns), start_ns, end_ns)
            if clipped is None:
                continue
            cs, ce = clipped
            raw_dur = max(1, int(e_ns) - int(s_ns))
            frac = (ce - cs) / raw_dur if int(e_ns) > int(s_ns) else 1.0
            mc_ivs.append((cs, ce, float(nbytes or 0) * frac))
    conn.close()

    # Aggregate kernel stats
    by_kname_ns: Counter[str] = Counter()
    by_kname_cnt: Counter[str] = Counter()
    glue_ns = 0
    glue_cnt = 0
    gemm_attn_ns = 0
    gemv_ns = 0
    weighted_occ = 0.0
    weighted_reg_util = 0.0
    weighted_regs = 0.0
    weighted_smem = 0.0
    weighted_l1_kib = 0.0
    weighted_wave_eff = 0.0
    wave_ns_denom = 0
    cliff_28k_no_tma_ns = 0
    total_spills = 0

    weighted_ldg_score = 0.0
    weighted_simt_score = 0.0
    sass_ns_denom = 0
    total_calls = 0
    mma_time_Counter: Counter[str] = Counter()

    for kname, cs, ce, grid_sz, blk_sz, regs, smem in k_events:
        dur = ce - cs
        by_kname_ns[kname] += dur
        by_kname_cnt[kname] += 1

        low_name = kname.lower()
        if any(w in low_name for w in GEMM_ATTN_KEYWORDS):
            gemm_attn_ns += dur
        if any(w in low_name for w in GEMV_DECODE_KEYWORDS):
            gemv_ns += dur

        occ = cuda_audit.compute_occupancy(arch, max(32, blk_sz), regs, smem)
        weighted_occ += dur * float(occ["warp_occupancy_pct"])
        weighted_reg_util += dur * float(occ["reg_utilization_pct"])
        weighted_regs += dur * regs
        weighted_smem += dur * smem
        l1_kib = float(occ["estimated_best_case_l1_kib"] or 64.0)
        weighted_l1_kib += dur * l1_kib

        if grid_sz > 0 and occ["ctas_per_sm"] > 0:
            full_wave = num_sms * occ["ctas_per_sm"]
            waves = grid_sz / full_wave
            wave_eff = 100.0 * (waves / math.ceil(waves))
            weighted_wave_eff += dur * wave_eff
            wave_ns_denom += dur

        ksass = match_kernel_entry(kname, fn_sass) or (agg_sass if has_sass else None)
        kres = match_kernel_entry(kname, res_map)
        if kres:
            total_spills += kres.get("spill_stores", 0) + kres.get("spill_loads", 0)

        if ksass and ksass["total_instructions"] > 0:
            weighted_ldg_score += dur * ksass["ldg_width_score_pct"]
            weighted_simt_score += dur * ksass["branchless_simt_score_pct"]
            sass_ns_denom += dur
            total_calls += ksass["call_ops"]
            mma_time_Counter[ksass["dominant_compute_unit"]] += dur
            if l1_kib <= 28.0 and ksass["ldg_counts"]["128b_or_tma"] == 0:
                cliff_28k_no_tma_ns += dur

    for kname, ns_sum in by_kname_ns.items():
        cnt = by_kname_cnt[kname]
        if (ns_sum / cnt) < 10_000 and cnt >= 32:  # < 10 us micro-kernel
            glue_ns += ns_sum
            glue_cnt += cnt

    denom = max(1, total_k_ns)
    avg_occ = weighted_occ / denom
    avg_reg_util = weighted_reg_util / denom
    avg_regs = weighted_regs / denom
    avg_smem_kib = (weighted_smem / denom) / 1024.0
    avg_l1_kib = weighted_l1_kib / denom
    glue_share_pct = 100.0 * glue_ns / denom
    gemm_share_pct = 100.0 * gemm_attn_ns / denom
    is_decode_like = (gemv_ns / denom) > 0.25 or (len(k_events) >= 200 and (total_k_ns / len(k_events)) < 40_000)

    # 1. Host RAM & PCIe Bus
    mc_merged = cuda_audit.merge_intervals([(s, e) for s, e, _ in mc_ivs])
    mc_tot_ns = sum(e - s for s, e in mc_merged)
    mc_ov_ns = cuda_audit.overlap_duration(mc_merged, compute_merged)
    mc_exp_ns = max(0, mc_tot_ns - mc_ov_ns)
    mc_bytes = sum(b for _, _, b in mc_ivs)
    mc_raw_ns = sum(e - s for s, e, _ in mc_ivs)
    mc_gbs = (mc_bytes / 1e9) / (mc_raw_ns / 1e9) if mc_raw_ns > 0 else 0.0

    if mc_tot_ns < 0.005 * max(1, compute_active_ns) and mc_exp_ns < 500_000:
        host_layer: dict[str, Any] = {
            "inactive": True, "efficiency_pct": 100.0, "time_weight_pct": 0.0,
            "headline": "VRAM-Resident (<0.5% PCIe)",
            "detail": f"Active GPU: {compute_active_ns / 1e6:.1f} ms • Host DMA: {mc_bytes / 1e6:.2f} MB",
            "tag": "[nsys]",
        }
    else:
        ov_pct = 100.0 * mc_ov_ns / max(1, mc_tot_ns)
        bw_pct = min(100.0, 100.0 * mc_gbs / max(1.0, pcie_ceiling_gbs))
        pcie_eff = round(0.65 * ov_pct + 0.35 * bw_pct, 1)
        pcie_wt = round(100.0 * mc_tot_ns / max(1, compute_active_ns + mc_exp_ns), 1)
        host_layer = {
            "efficiency_pct": pcie_eff, "time_weight_pct": pcie_wt,
            "headline": f"Overlap: {ov_pct:.1f}% • Link: {mc_gbs:.1f} GB/s",
            "detail": f"Exposed PCIe stall: {mc_exp_ns / 1e6:.1f} ms ({len(mc_ivs)} DMA calls, {mc_bytes / 1e6:.1f} MB)",
            "tag": "[nsys]",
            "diagnosis": f"{mc_exp_ns / 1e6:.1f} ms exposed PCIe stall ({ov_pct:.1f}% overlap, {mc_gbs:.1f} GB/s).",
            "action": "Use pinned memory + non-blocking staging stream w/ exp_done/h2d_done event lookahead.",
        }

    # 2. Global VRAM
    if "dram__throughput.avg.pct_of_peak_sustained_elapsed" in ncu:
        dram_pct = ncu["dram__throughput.avg.pct_of_peak_sustained_elapsed"]
        sec_req = ncu.get("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", 4.0)
        coal_pct = min(100.0, 400.0 / max(1.0, sec_req))
        vram_eff = round(0.5 * dram_pct + 0.5 * coal_pct, 1)
        vram_hl = f"DRAM Peak: {dram_pct:.0f}% • Coalescing: {coal_pct:.0f}%"
        vram_tag = "[ncu+nsys]"
    elif sass_ns_denom > 0:
        vram_eff = round(weighted_ldg_score / sass_ns_denom, 1)
        vram_hl = f"SASS Load-Width Coalescing Score: {vram_eff:.0f}%"
        vram_tag = "[SASS+nsys]"
    else:
        vram_eff = 75.0
        vram_hl = "Pass --binary/--sass-file or --ncu-csv for Coalescing %"
        vram_tag = "[nsys-only]"

    vram_wt = round(45.0 if is_decode_like else max(15.0, 100.0 - 0.7 * gemm_share_pct), 1)
    vram_layer = {
        "efficiency_pct": vram_eff, "time_weight_pct": vram_wt,
        "headline": vram_hl,
        "detail": f"Top kernel: {by_kname_ns.most_common(1)[0][0][:44]} ({100.0 * by_kname_ns.most_common(1)[0][1] / denom:.1f}%)",
        "tag": vram_tag,
        "diagnosis": f"Suboptimal global memory access width/coalescing ({vram_eff:.0f}%).",
        "action": "Vectorize to LDG.128 / TMA 2D or repack AoS quantized blocks into 128B-aligned Planar SoA.",
    }

    # 3. L2 Cache & Wave Quantization
    if "lts__t_sector_hit_rate.pct" in ncu:
        l2_eff = round(ncu["lts__t_sector_hit_rate.pct"], 1)
        l2_hl = f"L2 Sector Hit Rate: {l2_eff:.1f}%"
        l2_tag = "[ncu]"
    elif wave_ns_denom > 0:
        l2_eff = round(weighted_wave_eff / wave_ns_denom, 1)
        l2_hl = f"Grid Wave Quantization Eff: {l2_eff:.0f}% ({num_sms} SMs)"
        l2_tag = "[nsys-grid]"
    else:
        l2_eff = 80.0
        l2_hl = f"Target SKU: {num_sms} SMs"
        l2_tag = "[est]"

    l2_layer = {
        "efficiency_pct": l2_eff, "time_weight_pct": 14.0,
        "headline": l2_hl,
        "detail": "Check 2D CTA group swizzle (G=8/16) & LPT KV-head scheduling on long contexts",
        "tag": l2_tag,
        "diagnosis": f"Tail-wave quantization or low L2 tile locality ({l2_eff:.0f}%).",
        "action": "Align CTA grid count to SMs*CTAs_per_SM or apply 2D L2 group swizzling / LPT order.",
    }

    # 4. L1 & Shared Memory
    cliff_frac = cliff_28k_no_tma_ns / denom
    if "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum" in ncu:
        bconf = ncu["l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"]
        smem_eff = round(max(20.0, 95.0 - min(70.0, math.log10(max(1.0, bconf)) * 12.0)), 1)
        smem_tag = "[ncu+nsys]"
    else:
        smem_eff = round(max(35.0, 90.0 - 45.0 * cliff_frac), 1)
        smem_tag = "[SASS+nsys]" if has_sass else "[nsys]"

    smem_layer = {
        "efficiency_pct": smem_eff, "time_weight_pct": 22.0 if not is_decode_like else 10.0,
        "headline": f"Avg SMEM: {avg_smem_kib:.1f} KiB/CTA • Best-Case L1: {avg_l1_kib:.0f} KiB",
        "detail": (
            f"Warning: {100.0 * cliff_frac:.0f}% time in 28 KiB L1 cliff w/o TMA!"
            if cliff_frac > 0.1 else "Discrete SMEM carveout & L1 cache balanced"
        ),
        "tag": smem_tag,
        "diagnosis": f"SMEM carveout L1 cliff or bank conflict stall ({smem_eff:.0f}%).",
        "action": "Keep SMEM/SM <= 64 KiB for 64 KiB L1, or use TMA / cp.async.cg + 128B XOR swizzle.",
    }

    # 5. Register File, Occupancy & Launch Efficiency
    mean_k_us = (total_k_ns / max(1, len(k_events))) / 1e3
    drain_eff = 100.0 * mean_k_us / (mean_k_us + 1.8)
    spill_pen = 35.0 if total_spills > 0 else 0.0
    reg_occ_eff = round(max(20.0, min(96.0, 0.45 * drain_eff + 0.55 * min(100.0, avg_reg_util * 1.15) - spill_pen)), 1)
    if glue_share_pct >= 15.0:
        reg_occ_eff = round(min(reg_occ_eff, max(35.0, 85.0 - 1.3 * glue_share_pct)), 1)

    regs_layer = {
        "efficiency_pct": reg_occ_eff, "time_weight_pct": max(20.0, round(glue_share_pct + 12.0, 1)),
        "headline": f"Avg {avg_regs:.0f} Regs ({avg_reg_util:.0f}% RF) • Spills: {total_spills} • Glue: {glue_share_pct:.1f}%",
        "detail": f"{len(k_events):,} launches (mean {mean_k_us:.1f} us) • Warp Occ: {avg_occ:.1f}% ({glue_cnt:,} <10us launches)",
        "tag": "[nsys+res]" if res_map else "[nsys]",
        "diagnosis": (
            f"Confirmed register spills ({total_spills})" if total_spills > 0 else
            f"Micro-kernel launch storm ({glue_cnt:,} <10us calls = {glue_share_pct:.1f}% GPU time)"
        ),
        "action": (
            "Eliminate hot-loop STL/LDL spills via tile/register reduction." if total_spills > 0 else
            "Capture CUDA Graphs, fuse sequential micro-kernels, and enable PDL weight preload."
        ),
    }

    # 6. Warp SIMT & Schedulers
    if "smsp__thread_inst_executed_per_inst_executed.ratio" in ncu:
        simt_eff = round(100.0 * ncu["smsp__thread_inst_executed_per_inst_executed.ratio"] / 32.0, 1)
        simt_tag = "[ncu]"
    elif sass_ns_denom > 0:
        call_pen = min(30.0, total_calls * 0.1)
        simt_eff = round(max(25.0, (weighted_simt_score / sass_ns_denom) - call_pen), 1)
        simt_tag = "[SASS]"
    else:
        simt_eff = 82.0
        simt_tag = "[est]"

    simt_layer = {
        "efficiency_pct": simt_eff, "time_weight_pct": 18.0,
        "headline": f"SIMT / Branchless Score: {simt_eff:.0f}% • libm CALLs: {total_calls}",
        "detail": "Uniform @UP branches & SEL/LOP3/SHFL masks preserve 32-lane efficiency",
        "tag": simt_tag,
        "diagnosis": f"Warp divergence or un-inlined libm CALL overhead ({total_calls} CALLs).",
        "action": "Hoist ring/bounds checks to tile headers; replace tanhf/expf CALLs with PTX MUFU / exp2_fma_poly.",
    }

    # 7. Tensor Cores, ALU & SFU
    dom_tc = mma_time_Counter.most_common(1)[0][0] if mma_time_Counter else (
        "SIMT / GEMV Dominant" if is_decode_like else "GEMM / Attention Kernels"
    )
    if "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active" in ncu:
        tc_eff = round(ncu["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"], 1)
        tc_tag = "[ncu]"
    else:
        tc_eff = round(min(92.0, max(35.0, gemm_share_pct * (0.55 if is_decode_like else 1.05))), 1)
        tc_tag = "[SASS+nsys]" if has_sass else "[nsys]"

    compute_layer = {
        "efficiency_pct": tc_eff,
        "time_weight_pct": round(max(20.0, gemm_share_pct), 1),
        "expected_regime_bound": bool(is_decode_like and tc_eff < 60.0),
        "headline": f"{dom_tc[:38]} • GEMM/Attn: {gemm_share_pct:.1f}%",
        "detail": f"Total kernel sum: {total_k_ns / 1e6:.2f} ms across {len(by_kname_ns)} unique kernels",
        "tag": "[Expected @ Low-T]" if (is_decode_like and tc_eff < 60.0) else tc_tag,
        "diagnosis": f"Compute pipeline utilization is {tc_eff:.0f}% on {gemm_share_pct:.0f}% GEMM/Attn share.",
        "action": "Verify multi-stage TMA/cp.async overlap, sm_120 0x7F full-rate FP8 MMA, and cuBLASLt baseline.",
    }

    summary_line = (
        f"Active GPU: {compute_active_ns / 1e6:.1f} ms • {len(k_events):,} launches • GEMM/Attn: {gemm_share_pct:.1f}%"
    )
    layers = {
        "host_pcie": host_layer,
        "global_vram": vram_layer,
        "l2_cache": l2_layer,
        "smem_l1": smem_layer,
        "regs_occ": regs_layer,
        "warp_simt": simt_layer,
        "compute_cores": compute_layer,
    }
    return layers, summary_line


def cmd_auto_profile(args: argparse.Namespace) -> int:
    try:
        left_layers, left_sum = build_phase_from_telemetry(
            sqlite_path=args.left_sqlite,
            arch_input=args.arch,
            gpu_preset_name=args.gpu,
            pcie_ceiling_gbs=args.pcie_gbs,
            device_filter=args.device,
            start_ns=args.left_start_ns,
            end_ns=args.left_end_ns,
            binary=args.left_binary or args.binary,
            sass_file=args.left_sass or args.sass_file,
            res_file=args.left_res or args.res_file,
            ncu_csv=args.left_ncu_csv,
        )
        right_sqlite = args.right_sqlite or args.left_sqlite
        right_layers, right_sum = build_phase_from_telemetry(
            sqlite_path=right_sqlite,
            arch_input=args.arch,
            gpu_preset_name=args.gpu,
            pcie_ceiling_gbs=args.pcie_gbs,
            device_filter=args.device,
            start_ns=args.right_start_ns,
            end_ns=args.right_end_ns,
            binary=args.right_binary or args.binary,
            sass_file=args.right_sass or args.sass_file,
            res_file=args.right_res or args.res_file,
            ncu_csv=args.right_ncu_csv,
        )
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Error building auto-profile: {exc}", file=sys.stderr)
        return 1

    if args.focus:
        focus_targets = [x.strip() for x in args.focus.split(",") if x.strip()]
    else:
        # Automatically highlight layers whose efficiency shifted by >= 5 percentage points!
        focus_targets = [
            k for k, _, _ in LAYER_ORDER
            if abs(float(right_layers[k]["efficiency_pct"]) - float(left_layers[k]["efficiency_pct"])) >= 5.0
        ]

    default_left_lbl = "BEFORE (Baseline)" if args.mode == "commit-focus" else "PREFILL"
    default_right_lbl = "AFTER (Optimized)" if args.mode == "commit-focus" else "DECODE"

    profile: dict[str, Any] = {
        "mode": args.mode,
        "title": args.title or (
            "CUDA Hierarchy Commit Focus Heatmap" if args.mode == "commit-focus"
            else "CUDA Hierarchy Efficiency: Prefill vs Decode"
        ),
        "subtitle": args.subtitle or f"Auto-Profiled ({args.gpu}, {args.arch}) from nsys SQLite"
                    + (" + SASS" if (args.binary or args.left_binary or args.sass_file or args.left_sass) else "")
                    + (" + Nsight Compute" if (args.left_ncu_csv or args.right_ncu_csv) else ""),
        "footer": "Green (>=80%): Near Ceiling  •  Amber (55–79%): Headroom  •  Red (<55%): Bottleneck  •  Slate: Inactive",
        "focus_targets": focus_targets,
        "focus_banner": args.banner or (
            f"Auto-Detected Focus Layers (>=5pt shift): {', '.join(focus_targets) or 'None'}"
            if args.mode == "commit-focus" else
            "Automated Telemetry Comparison Across 7 CUDA Hardware Tiers"
        ),
        "left_label": args.left_label or default_left_lbl,
        "right_label": args.right_label or default_right_lbl,
        "left_summary": left_sum,
        "right_summary": right_sum,
        "left": left_layers,
        "right": right_layers,
    }

    if args.profile_out:
        p_out = Path(args.profile_out)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        p_out.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    svg_content = render_hierarchy_svg(profile)
    svg_out = Path(args.svg)
    svg_out.parent.mkdir(parents=True, exist_ok=True)
    svg_out.write_text(svg_content, encoding="utf-8")

    if args.report_json:
        bots = {
            "left_bottlenecks": infer_bottlenecks(profile["left_label"], left_layers),
            "right_bottlenecks": infer_bottlenecks(profile["right_label"], right_layers),
        }
        Path(args.report_json).write_text(json.dumps(bots, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    png_status = "not requested"
    if args.png:
        ok = try_rasterize_svg_to_png(svg_out, Path(args.png), width_px=args.width)
        png_status = args.png if ok else "skipped (no system SVG rasterizer found)"

    print(f"Success! Auto-profile SVG: {svg_out} | PNG: {png_status}")
    return 0


# ---------------------------------------------------------------------------
# CLI Subcommands
# ---------------------------------------------------------------------------
def cmd_sass_metrics(args: argparse.Namespace) -> int:
    if args.binary:
        cmd = ["cuobjdump", "-sass"]
        if args.function_filter:
            cmd.extend(["-fun", args.function_filter])
        cmd.append(args.binary)
        sass_text = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    elif args.sass_file:
        sass_text = Path(args.sass_file).read_text(encoding="utf-8", errors="replace")
    else:
        print("Error: specify either --binary or --sass-file.", file=sys.stderr)
        return 1

    metrics = extract_sass_hierarchy_metrics(sass_text)
    return cuda_audit.emit_output(metrics, args.output)


def cmd_render(args: argparse.Namespace) -> int:
    if args.input:
        profile = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        profile = get_preset_profile(args.preset)

    if args.focus:
        profile["focus_targets"] = [x.strip() for x in args.focus.split(",") if x.strip()]

    svg_content = render_hierarchy_svg(profile)
    svg_out = Path(args.svg)
    svg_out.parent.mkdir(parents=True, exist_ok=True)
    svg_out.write_text(svg_content, encoding="utf-8")

    bottlenecks = {
        "left_bottlenecks": infer_bottlenecks(profile["left_label"], profile["left"]),
        "right_bottlenecks": infer_bottlenecks(profile["right_label"], profile["right"]),
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(bottlenecks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    png_status = "not requested"
    if args.png:
        png_out = Path(args.png)
        ok = try_rasterize_svg_to_png(svg_out, png_out, width_px=args.width)
        png_status = str(png_out) if ok else "skipped (no system SVG rasterizer found; attach SVG directly to GitHub PR)"

    print(f"Success! SVG written to: {svg_out} | PNG: {png_status}")
    return 0


def cmd_dump_template(args: argparse.Namespace) -> int:
    prof = get_preset_profile(args.preset)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prof, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Success! Template JSON written to: {out}")
    return 0
