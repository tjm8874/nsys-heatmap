"""Core Nsight Systems SQLite, SASS/Occupancy, Roofline & ABBA analysis primitives."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RESERVED_SMEM_PER_CTA_BYTES = 1024

ARCH_SPECS: dict[str, dict[str, Any]] = {
    "sm_80": {
        "regs_per_sm": 65536, "max_regs_thread": 255, "max_threads_cta": 1024,
        "max_warps_sm": 64, "max_ctas_sm": 32,
        "l1_smem_kb": 192, "max_smem_sm_kb": 164, "max_smem_cta_kb": 163,
        "carveouts_kb": [0, 8, 16, 32, 64, 100, 132, 164], "subparts": 4,
    },
    "sm_89": {
        "regs_per_sm": 65536, "max_regs_thread": 255, "max_threads_cta": 1024,
        "max_warps_sm": 48, "max_ctas_sm": 24,
        "l1_smem_kb": 128, "max_smem_sm_kb": 100, "max_smem_cta_kb": 99,
        "carveouts_kb": [0, 8, 16, 32, 64, 100], "subparts": 4,
    },
    "sm_90a": {
        "regs_per_sm": 65536, "max_regs_thread": 255, "max_threads_cta": 1024,
        "max_warps_sm": 64, "max_ctas_sm": 32,
        "l1_smem_kb": 256, "max_smem_sm_kb": 228, "max_smem_cta_kb": 227,
        "carveouts_kb": [0, 8, 16, 32, 64, 100, 132, 164, 196, 228], "subparts": 4,
    },
    "sm_100a": {
        "regs_per_sm": 65536, "max_regs_thread": 255, "max_threads_cta": 1024,
        "max_warps_sm": 64, "max_ctas_sm": 32,
        "l1_smem_kb": 256, "max_smem_sm_kb": 228, "max_smem_cta_kb": 227,
        "carveouts_kb": [0, 8, 16, 32, 64, 100, 132, 164, 196, 228], "subparts": 4,
    },
    "sm_120": {
        "regs_per_sm": 65536, "max_regs_thread": 255, "max_threads_cta": 1024,
        "max_warps_sm": 48, "max_ctas_sm": 32,
        "l1_smem_kb": 128, "max_smem_sm_kb": 100, "max_smem_cta_kb": 99,
        "carveouts_kb": [0, 8, 16, 32, 64, 100], "subparts": 4,
    },
}

ARCH_ALIASES: dict[str, str] = {
    "sm_80": "sm_80", "sm_86": "sm_89", "sm_89": "sm_89",
    "sm_90": "sm_90a", "sm_90a": "sm_90a",
    "sm_100": "sm_100a", "sm_100a": "sm_100a", "sm_100f": "sm_100a", "sm_103": "sm_100a",
    "sm_120": "sm_120", "sm_120a": "sm_120", "sm_120f": "sm_120", "sm_121": "sm_120", "sm_121a": "sm_120",
}

GPU_PRESETS: dict[str, dict[str, Any]] = {
    "a100_sxm_400w": {
        "arch": "sm_80", "sms": 108, "dram_gbs": 2039.0,
        "provenance": "estimated sustained ceiling (A100 SXM4 80GB HBM2e @ 400W)",
        "tflops": {"fp32": 19.0, "fp16": 270.0, "bf16": 270.0, "int8": 540.0, "fp8": None, "mxfp8": None, "nvfp4": None},
    },
    "rtx_4090_450w": {
        "arch": "sm_89", "sms": 128, "dram_gbs": 1008.0,
        "provenance": "estimated sustained ceiling (GeForce RTX 4090 24GB GDDR6X @ 450W; FP32-accum caps apply)",
        "tflops": {"fp32": 75.0, "fp16": 165.0, "bf16": 165.0, "int8": 600.0, "fp8": 330.0, "mxfp8": None, "nvfp4": None},
    },
    "rtx_pro_5000_300w": {
        "arch": "sm_120", "sms": 110, "dram_gbs": 1344.0,
        "provenance": "empirical sustained ceiling (RTX PRO 5000 72GB GDDR7 @ 300W, Driver 595.99, CUDA 13.4)",
        "tflops": {"fp32": 48.0, "fp16": 155.0, "bf16": 155.0, "int8": 250.0, "fp8": 310.0, "mxfp8": 310.0, "nvfp4": 510.0},
    },
    "rtx_pro_4500_200w": {
        "arch": "sm_120", "sms": 84, "dram_gbs": 896.0,
        "provenance": "estimated power-scaled ceiling (RTX PRO 4500 32GB GDDR7 @ 200W); calibrate w/ local cublasLt",
        "tflops": {"fp32": 36.0, "fp16": 118.0, "bf16": 118.0, "int8": 195.0, "fp8": 235.0, "mxfp8": 235.0, "nvfp4": 390.0},
    },
    "rtx_5090_575w": {
        "arch": "sm_120", "sms": 170, "dram_gbs": 1792.0,
        "provenance": "estimated sustained ceiling (GeForce RTX 5090 32GB GDDR7 @ 575W; SKU FP32-accum caps may apply)",
        "tflops": {"fp32": 95.0, "fp16": 330.0, "bf16": 330.0, "int8": 560.0, "fp8": 660.0, "mxfp8": 660.0, "nvfp4": 1150.0},
    },
    "rtx_5070_ti_300w": {
        "arch": "sm_120", "sms": 70, "dram_gbs": 896.0,
        "provenance": "estimated sustained ceiling (GeForce RTX 5070 Ti 16GB GDDR7 @ 300W)",
        "tflops": {"fp32": 38.0, "fp16": 125.0, "bf16": 125.0, "int8": 210.0, "fp8": 250.0, "mxfp8": 250.0, "nvfp4": 440.0},
    },
    "h100_sxm_700w": {
        "arch": "sm_90a", "sms": 132, "dram_gbs": 3350.0,
        "provenance": "empirical sustained ceiling (~80% of boost, H100 SXM5 HBM3 @ 700W)",
        "tflops": {"fp32": 60.0, "fp16": 800.0, "bf16": 800.0, "int8": 1500.0, "fp8": 1500.0, "mxfp8": None, "nvfp4": None},
    },
    "b200_sxm_1000w": {
        "arch": "sm_100a", "sms": 148, "dram_gbs": 8000.0,
        "provenance": "estimated sustained ceiling (B200 SXM HBM3e @ 1000W)",
        "tflops": {"fp32": 80.0, "fp16": 2000.0, "bf16": 2000.0, "int8": 3600.0, "fp8": 3800.0, "mxfp8": 3800.0, "nvfp4": 7500.0},
    },
}


def normalize_arch(arch: str) -> str:
    norm = ARCH_ALIASES.get(arch.lower().strip())
    if norm is None:
        raise ValueError(f"Unsupported architecture '{arch}'. Supported: {sorted(ARCH_ALIASES.keys())}")
    return norm


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    pts = sorted(intervals, key=lambda x: x[0])
    merged = [pts[0]]
    for s, e in pts[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def overlap_duration(a_merged: list[tuple[int, int]], b_merged: list[tuple[int, int]]) -> int:
    i = j = 0
    total = 0
    while i < len(a_merged) and j < len(b_merged):
        s = max(a_merged[i][0], b_merged[j][0])
        e = min(a_merged[i][1], b_merged[j][1])
        if s < e:
            total += e - s
        if a_merged[i][1] < b_merged[j][1]:
            i += 1
        else:
            j += 1
    return total


def clip_interval(start_ns: int, end_ns: int, win_start: int | None, win_end: int | None) -> tuple[int, int] | None:
    cs = max(start_ns, win_start) if win_start is not None else start_ns
    ce = min(end_ns, win_end) if win_end is not None else end_ns
    if ce <= cs:
        return None
    return cs, ce


def emit_output(payload: dict[str, Any], output_path: str | None) -> int:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        print(f"Success! Report written to: {out}")
    else:
        print(rendered)
    return 0


def cmd_nsys_top(args: argparse.Namespace) -> int:
    db_path = Path(args.sqlite)
    if not db_path.is_file():
        print(f"Error: SQLite file not found: {db_path}", file=sys.stderr)
        return 1
    if args.start_ns is not None and args.end_ns is not None and args.end_ns <= args.start_ns:
        print("Error: --end-ns must be strictly greater than --start-ns.", file=sys.stderr)
        return 1

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    warnings: list[str] = []

    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
        g_cnt = cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchone()[0]
        if g_cnt > 0:
            warnings.append(
                f"CUDA GRAPH TRACE DETECTED ({g_cnt} graph executions): Default `--cuda-graph-trace=graph` hides "
                "individual kernels inside graphs from CUPTI_ACTIVITY_KIND_KERNEL. Capture a separate profile with "
                "`--cuda-graph-trace=node` for intra-graph kernel breakdown (and keep node-trace OFF for wall-time runs)."
            )

    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        conn.close()
        print("Error: CUPTI_ACTIVITY_KIND_KERNEL table missing in SQLite export.", file=sys.stderr)
        return 1

    name_col = "demangledName" if args.demangled else "shortName"
    str_table = "StringIds" if "StringIds" in tables else None

    where_clauses: list[str] = []
    params: list[Any] = []
    if args.start_ns is not None:
        where_clauses.append("k.end > ?")
        params.append(args.start_ns)
    if args.end_ns is not None:
        where_clauses.append("k.start < ?")
        params.append(args.end_ns)
    if args.device is not None:
        where_clauses.append("k.deviceId = ?")
        params.append(args.device)
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    if str_table:
        q = f"""
            SELECT COALESCE(s.value, CAST(k.{name_col} AS TEXT)) AS kname,
                   k.start, k.end, k.deviceId,
                   k.registersPerThread, k.dynamicSharedMemory + k.staticSharedMemory AS smem
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON k.{name_col} = s.id
            {where_sql}
        """
    else:
        q = f"""
            SELECT CAST(k.{name_col} AS TEXT), k.start, k.end, k.deviceId,
                   k.registersPerThread, k.dynamicSharedMemory + k.staticSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            {where_sql}
        """

    kernel_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "ns": 0, "count": 0, "min_ns": 10**18, "max_ns": 0,
            "regs": set(), "smem": set(), "devices": set(),
        }
    )
    compute_by_dev: dict[int, list[tuple[int, int]]] = defaultdict(list)
    kernel_ns_by_dev: dict[int, int] = defaultdict(int)
    launches_by_dev: dict[int, int] = defaultdict(int)
    total_kernel_ns = 0
    total_launches = 0

    for kname, s_ns, e_ns, dev_id, regs, smem in cur.execute(q, params):
        raw_s, raw_e = int(s_ns), int(e_ns)
        clipped = clip_interval(raw_s, raw_e, args.start_ns, args.end_ns)
        if clipped is None:
            continue
        cs, ce = clipped
        dur = ce - cs
        dev = int(dev_id or 0)

        total_kernel_ns += dur
        total_launches += 1
        compute_by_dev[dev].append((cs, ce))
        kernel_ns_by_dev[dev] += dur
        launches_by_dev[dev] += 1

        st = kernel_stats[kname]
        st["ns"] += dur
        st["count"] += 1
        st["min_ns"] = min(st["min_ns"], dur)
        st["max_ns"] = max(st["max_ns"], dur)
        st["devices"].add(dev)
        if regs is not None:
            st["regs"].add(int(regs))
        if smem is not None:
            st["smem"].add(int(smem))

    merged_compute_by_dev = {dev: merge_intervals(ivs) for dev, ivs in compute_by_dev.items()}

    memcpy_by_dev: dict[int, dict[str, list[tuple[int, int, float]]]] = defaultdict(lambda: defaultdict(list))
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        mc_cols = {r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
        dev_expr = "m.deviceId" if "deviceId" in mc_cols else "0"
        mc_where = where_sql.replace("k.", "m.") if ("deviceId" in mc_cols or args.device is None) else (
            " AND ".join(c.replace("k.", "m.") for c in where_clauses if "deviceId" not in c)
        )
        if mc_where and not mc_where.startswith("WHERE "):
            mc_where = "WHERE " + mc_where
        mc_params = params if ("deviceId" in mc_cols or args.device is None) else [
            p for c, p in zip(where_clauses, params) if "deviceId" not in c
        ]

        mq = f"SELECT m.copyKind, m.bytes, m.start, m.end, {dev_expr} FROM CUPTI_ACTIVITY_KIND_MEMCPY m {mc_where}"
        kind_map = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
        for ckind, nbytes, s_ns, e_ns, dev_id in cur.execute(mq, mc_params):
            raw_s, raw_e = int(s_ns), int(e_ns)
            clipped = clip_interval(raw_s, raw_e, args.start_ns, args.end_ns)
            if clipped is None:
                continue
            cs, ce = clipped
            raw_dur = max(1, raw_e - raw_s)
            frac = (ce - cs) / raw_dur if raw_e > raw_s else 1.0
            scaled_bytes = float(nbytes or 0) * frac
            dev = int(dev_id or 0)
            label = kind_map.get(int(ckind), f"Kind_{ckind}")
            memcpy_by_dev[dev][label].append((cs, ce, scaled_bytes))

    all_devs = sorted(set(merged_compute_by_dev.keys()) | set(memcpy_by_dev.keys()))
    per_device_summary: dict[str, Any] = {}

    for dev in all_devs:
        dev_compute = merged_compute_by_dev.get(dev, [])
        dev_active_ns = sum(e - s for s, e in dev_compute)
        dev_mc_out: dict[str, Any] = {}
        for label, rows in sorted(memcpy_by_dev.get(dev, {}).items()):
            ivs = merge_intervals([(s, e) for s, e, _ in rows])
            tot_ns = sum(e - s for s, e in ivs)
            ov_ns = overlap_duration(ivs, dev_compute)
            tot_bytes = sum(b for _, _, b in rows)
            raw_sum_ns = sum(e - s for s, e, _ in rows)
            dev_mc_out[label] = {
                "calls": len(rows),
                "window_mb": round(tot_bytes / 1e6, 3),
                "merged_ms": round(tot_ns / 1e6, 3),
                "overlapped_with_same_gpu_compute_ms": round(ov_ns / 1e6, 3),
                "overlap_pct": round(100.0 * ov_ns / tot_ns, 2) if tot_ns > 0 else 0.0,
                "exposed_stall_ms": round((tot_ns - ov_ns) / 1e6, 3),
                "effective_gbs": round((tot_bytes / 1e9) / (raw_sum_ns / 1e9), 2) if raw_sum_ns > 0 else 0.0,
            }
            if label == "HtoD" and dev_mc_out[label]["exposed_stall_ms"] > 10.0:
                h = dev_mc_out[label]
                warnings.append(
                    f"GPU {dev} EXPOSED PCIe HtoD STALL: {h['exposed_stall_ms']} ms unhidden "
                    f"({h['overlap_pct']}% overlapped with GPU {dev} compute). "
                    "Ensure pinned host memory and add non-blocking staging stream + event lookahead."
                )

        per_device_summary[f"gpu_{dev}"] = {
            "device_id": dev,
            "kernel_sum_ms": round(kernel_ns_by_dev.get(dev, 0) / 1e6, 3),
            "merged_compute_active_ms": round(dev_active_ns / 1e6, 3),
            "launches": launches_by_dev.get(dev, 0),
            "memcpy_overlap": dev_mc_out,
        }

    ranked = sorted(kernel_stats.items(), key=lambda kv: kv[1]["ns"], reverse=True)
    top_rows = []
    glue_launches = 0
    glue_ns = 0
    for _name, st in ranked:
        mean_us = (st["ns"] / st["count"]) / 1e3
        if mean_us < 10.0 and st["count"] >= 64:
            glue_launches += st["count"]
            glue_ns += st["ns"]

    for rank, (name, st) in enumerate(ranked[: args.limit], start=1):
        top_rows.append({
            "rank": rank,
            "kernel": name,
            "devices": sorted(st["devices"]),
            "total_ms": round(st["ns"] / 1e6, 3),
            "share_pct": round(100.0 * st["ns"] / total_kernel_ns, 2) if total_kernel_ns > 0 else 0.0,
            "launches": st["count"],
            "mean_us": round((st["ns"] / st["count"]) / 1e3, 2),
            "min_us": round(st["min_ns"] / 1e3, 2),
            "max_us": round(st["max_ns"] / 1e3, 2),
            "regs": sorted(st["regs"]),
            "smem_kib": [round(x / 1024.0, 2) for x in sorted(st["smem"])],
        })

    if total_kernel_ns > 0 and (glue_ns / total_kernel_ns) >= 0.10:
        warnings.append(
            f"GLUE KERNEL STORM: {glue_launches} launches of <10 us micro-kernels consume "
            f"{glue_ns / 1e6:.2f} ms ({100.0 * glue_ns / total_kernel_ns:.1f}% of total kernel time). "
            "Evaluate CUDA Graphs, producer-consumer kernel fusion, or Programmatic Dependent Launch (PDL)."
        )

    report = {
        "sqlite": str(db_path),
        "window_ns": {"start_ns": args.start_ns, "end_ns": args.end_ns, "clipped": args.start_ns is not None or args.end_ns is not None},
        "total_kernel_sum_ms": round(total_kernel_ns / 1e6, 3),
        "total_launches": total_launches,
        "per_device": per_device_summary,
        "warnings": warnings,
        "top_kernels": top_rows,
    }
    conn.close()
    return emit_output(report, args.output)


def compute_occupancy(arch_input: str, threads: int, regs: int, smem_bytes: int) -> dict[str, Any]:
    arch_name = normalize_arch(arch_input)
    spec = ARCH_SPECS[arch_name]
    max_smem_cta_bytes = spec["max_smem_cta_kb"] * 1024
    max_smem_sm_bytes = spec["max_smem_sm_kb"] * 1024

    reasons: list[str] = []
    if threads <= 0 or threads > spec["max_threads_cta"]:
        reasons.append("threads_exceed_cta_max")
    if regs > spec["max_regs_thread"]:
        reasons.append("regs_exceed_thread_max")
    if smem_bytes < 0 or smem_bytes > max_smem_cta_bytes:
        reasons.append("smem_exceeds_cta_max")

    warps_per_cta = max(0, math.ceil(threads / 32)) if threads > 0 else 0
    per_cta_smem_with_rsvd = smem_bytes + RESERVED_SMEM_PER_CTA_BYTES
    if per_cta_smem_with_rsvd > max_smem_sm_bytes and "smem_exceeds_cta_max" not in reasons:
        reasons.append("smem_exceeds_cta_max")

    if reasons:
        return {
            "arch": arch_name,
            "launchable": False,
            "threads_per_cta": threads,
            "warps_per_cta": warps_per_cta,
            "ctas_per_sm": 0,
            "active_warps_per_sm": 0,
            "warp_occupancy_pct": 0.0,
            "reg_utilization_pct": 0.0,
            "selected_smem_carveout_kib": None,
            "estimated_best_case_l1_kib": None,
            "limiter": "+".join(reasons),
            "note": "Unlaunchable CTA configuration! Exceeds hardware per-block limits.",
        }

    alloc_regs_thread = max(1, math.ceil(regs / 8) * 8) if regs > 0 else 32
    regs_per_cta = warps_per_cta * 32 * alloc_regs_thread

    by_warps = spec["max_warps_sm"] // warps_per_cta
    by_regs = spec["regs_per_sm"] // regs_per_cta if regs_per_cta > 0 else spec["max_ctas_sm"]
    by_smem = max_smem_sm_bytes // per_cta_smem_with_rsvd

    ctas_per_sm = max(0, min(spec["max_ctas_sm"], by_warps, by_regs, by_smem))
    active_warps = ctas_per_sm * warps_per_cta

    needed_smem_bytes = ctas_per_sm * per_cta_smem_with_rsvd
    carveout_kib = min((c for c in spec["carveouts_kb"] if c * 1024 >= needed_smem_bytes), default=None)
    best_case_l1_kib = float(spec["l1_smem_kb"] - carveout_kib) if carveout_kib is not None else 0.0

    limiter = (
        "registers" if by_regs <= min(by_warps, by_smem) else
        "smem" if by_smem <= min(by_warps, by_regs) else
        "warps_or_cta_cap"
    )
    return {
        "arch": arch_name,
        "launchable": ctas_per_sm > 0,
        "threads_per_cta": threads,
        "warps_per_cta": warps_per_cta,
        "ctas_per_sm": ctas_per_sm,
        "active_warps_per_sm": active_warps,
        "warp_occupancy_pct": round(100.0 * active_warps / spec["max_warps_sm"], 1),
        "reg_utilization_pct": round(100.0 * (ctas_per_sm * regs_per_cta) / spec["regs_per_sm"], 1),
        "selected_smem_carveout_kib": carveout_kib,
        "estimated_best_case_l1_kib": best_case_l1_kib,
        "limiter": limiter,
        "note": (
            "Includes 1 KiB/CTA reserved SMEM and discrete carveout steps; cross-check on target GPU via "
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor and ncu 'Shared Memory Configuration Size'."
        ),
    }


def classify_ldg_widths(ops: Counter[str]) -> tuple[int, int]:
    narrow_8_16 = 0
    wide_128_256 = 0
    for op, cnt in ops.items():
        if not op.startswith(("LDG.", "LDGSTS.")):
            continue
        if re.search(r"\.(?:[US]?8|[US]?16)(?:\.|$)", op):
            narrow_8_16 += cnt
        elif re.search(r"\.(?:128|256)(?:\.|$)", op):
            wide_128_256 += cnt
    return narrow_8_16, wide_128_256


def classify_tensor_core_mma(ops: Counter[str]) -> tuple[int, dict[str, int]]:
    by_family: Counter[str] = Counter()
    total = 0
    for op, cnt in ops.items():
        if "MMA" in op and not op.startswith("MMAC"):
            root = op.split(".")[0]
            by_family[root] += cnt
            total += cnt
    return total, dict(sorted(by_family.items()))


def cmd_sass_audit(args: argparse.Namespace) -> int:
    try:
        arch_norm = normalize_arch(args.arch)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    sass_text = ""
    res_text = ""
    if args.binary:
        bin_path = Path(args.binary)
        if not bin_path.is_file():
            print(f"Error: binary not found: {bin_path}", file=sys.stderr)
            return 1
        try:
            res_text = subprocess.check_output(["cuobjdump", "-res-usage", str(bin_path)], text=True, stderr=subprocess.STDOUT)
            sass_cmd = ["cuobjdump", "-sass"]
            if args.function_filter:
                sass_cmd.extend(["-fun", args.function_filter])
            sass_cmd.append(str(bin_path))
            sass_text = subprocess.check_output(sass_cmd, text=True, stderr=subprocess.STDOUT)
        except FileNotFoundError:
            print("Error: `cuobjdump` not found on PATH. Pass --sass-file / --res-file on machines without CUDA toolkit.", file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as exc:
            print(f"Error running cuobjdump:\n{exc.output}", file=sys.stderr)
            return 1
    else:
        if args.sass_file:
            sass_text = Path(args.sass_file).read_text(encoding="utf-8", errors="replace")
        if args.res_file:
            res_text = Path(args.res_file).read_text(encoding="utf-8", errors="replace")

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

    fn_opcodes: dict[str, Counter[str]] = defaultdict(Counter)
    active_fn: str | None = None
    inst_re = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?U?P(?:T|\d+)\s+)?([A-Z0-9_.]+)")
    for line in sass_text.splitlines():
        m_hdr = re.search(r"Function\s*:\s*([^\s]+)", line)
        if m_hdr:
            active_fn = m_hdr.group(1)
            continue
        if active_fn:
            m_op = inst_re.search(line)
            if m_op:
                fn_opcodes[active_fn][m_op.group(1)] += 1

    all_fns = sorted(set(res_map.keys()) | set(fn_opcodes.keys()))
    if args.function_filter:
        all_fns = [f for f in all_fns if args.function_filter in f]

    audited_kernels = []
    for fn in all_fns:
        ops = fn_opcodes.get(fn, Counter())
        rmeta = res_map.get(fn, {"regs": args.regs or 0, "smem_static": 0, "stack": 0, "spill_stores": 0, "spill_loads": 0})
        regs = args.regs if args.regs is not None else rmeta["regs"]
        smem = args.dynamic_smem + rmeta["smem_static"]

        calls = sum(c for k, c in ops.items() if k.startswith("CALL"))
        stl_ldl = sum(c for k, c in ops.items() if k.startswith(("STL", "LDL")))
        narrow_ldg, wide_ldg = classify_ldg_widths(ops)
        tma_ops = sum(c for k, c in ops.items() if "UTMALDG" in k or "UTMASTG" in k)
        prmt_ops = sum(c for k, c in ops.items() if k.startswith("PRMT"))
        mufu_ops = sum(c for k, c in ops.items() if k.startswith("MUFU"))
        mma_ops, mma_by_family = classify_tensor_core_mma(ops)

        findings: list[str] = []
        verification_steps: list[str] = []

        if calls > 0:
            findings.append(
                f"CRITICAL: {calls} subroutine CALL instructions detected. Check for un-inlined libm `tanhf`/`expf`/`powf` "
                "and consider inline PTX `tanh.approx.f32` / `ex2.approx.ftz.f32` (validate changed numerics via quality gate)."
            )

        has_confirmed_spill = rmeta["spill_stores"] > 0 or rmeta["spill_loads"] > 0
        if has_confirmed_spill:
            findings.append(
                f"CRITICAL: Confirmed register spill in res-usage (SPILL_STORES={rmeta['spill_stores']}, "
                f"SPILL_LOADS={rmeta['spill_loads']}, STL+LDL={stl_ldl}, STACK={rmeta['stack']}B)."
            )
            verification_steps.append(
                "Compile with `-lineinfo -Xptxas -v` and run `cuobjdump -sass -lineinfo` (or `ncu --page source`) "
                "to verify whether spill STL/LDL instructions execute inside the hot K/KV loop vs cold cleanup paths."
            )
        elif stl_ldl > 0 or rmeta["stack"] > 0:
            findings.append(
                f"INVESTIGATE LOCAL MEMORY: STL+LDL={stl_ldl}, STACK={rmeta['stack']}B, with SPILL_STORES=0. "
                "Often caused by dynamically indexed local arrays or ABI stack frames rather than register spills."
            )
            verification_steps.append(
                "Before declaring a register spill: (1) check `nvcc -Xptxas -v` `spill stores / spill loads`, "
                "(2) map `STL`/`LDL` to C++ lines via `-lineinfo` (look for non-unrolled local arrays `T arr[i]`), "
                "and (3) check hot-path execution count (`ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum`)."
            )

        if narrow_ldg > wide_ldg and narrow_ldg >= 16:
            findings.append(
                f"HIGH: {narrow_ldg} narrow 8/16-bit global loads vs {wide_ldg} 128/256-bit loads. "
                "Coalesce into 32/64/128-bit loads (`int`/`int2`/`int4`) or repack AoS blocks into planar SoA."
            )
        if prmt_ops >= 128:
            findings.append(
                f"MEDIUM: {prmt_ops} PRMT byte-permute instructions. On sm_100+/sm_120a, evaluate replacing LUT unpacking "
                "with native `cvt.rn.f16x2.e2m1x2` (`F2FP`) or `__nv_cvt_fp8x2_to_halfraw2`."
            )

        occ = compute_occupancy(arch_norm, args.threads, regs, smem) if args.threads > 0 else None
        if occ:
            if not occ["launchable"]:
                findings.append(
                    f"UNLAUNCHABLE CONFIG: {occ['limiter']} (threads={args.threads}, regs={regs}, smem={smem}B). "
                    "Kernel launch will fail with cudaErrorInvalidValue / cudaErrorLaunchOutOfResources."
                )
            elif (
                arch_norm in ("sm_89", "sm_120")
                and occ["estimated_best_case_l1_kib"] is not None
                and occ["estimated_best_case_l1_kib"] <= 28.0
                and tma_ops == 0
            ):
                findings.append(
                    f"DISCRETE SMEM CARVEOUT CLIFF: Carveout jumped to {occ['selected_smem_carveout_kib']} KiB, leaving at most "
                    f"{occ['estimated_best_case_l1_kib']} KiB L1 cache with 0 TMA instructions. Keep total SMEM/SM (incl. 1 KiB/CTA "
                    "reserved) <= 64 KiB to retain 64 KiB L1, or use `cp.async.cg` / TMA so streaming reads bypass L1."
                )

        audited_kernels.append({
            "function": fn,
            "resources": {**rmeta, "total_smem_bytes": smem},
            "occupancy": occ,
            "instruction_summary": {
                "total_instructions": sum(ops.values()),
                "CALL": calls,
                "STL_LDL": stl_ldl,
                "narrow_LDG_8_16": narrow_ldg,
                "wide_LDG_128_256_LDGSTS": wide_ldg,
                "TMA_UTMA": tma_ops,
                "MMA_TensorCore": mma_ops,
                "MMA_by_family": mma_by_family,
                "MUFU_SFU": mufu_ops,
                "PRMT": prmt_ops,
            },
            "findings": findings,
            "verification_steps": verification_steps,
        })

    return emit_output({"arch": arch_norm, "kernels": audited_kernels}, args.output)


def cmd_roofline(args: argparse.Namespace) -> int:
    preset = GPU_PRESETS.get(args.gpu)
    peak_gbs = args.peak_gbs if args.peak_gbs is not None else (preset["dram_gbs"] if preset else None)
    if args.peak_tflops is not None:
        peak_tflops = args.peak_tflops
    elif preset:
        peak_tflops = preset["tflops"].get(args.dtype)
        if peak_tflops is None:
            print(
                f"Error: dtype '{args.dtype}' is not natively supported on preset '{args.gpu}' "
                f"(arch {preset['arch']}). Choose a supported dtype or pass --peak-tflops explicitly.",
                file=sys.stderr,
            )
            return 1
    else:
        peak_tflops = None

    if not peak_gbs or not peak_tflops:
        print("Error: specify a valid --gpu preset or pass both --peak-gbs and --peak-tflops.", file=sys.stderr)
        return 1

    sec = args.time_us * 1e-6
    algo_gbs = (args.bytes / 1e9) / sec if sec > 0 else 0.0
    achieved_tflops = (args.flops / 1e12) / sec if sec > 0 else 0.0
    ai = args.flops / args.bytes if args.bytes > 0 else float("inf")
    ridge_ai = (peak_tflops * 1e12) / (peak_gbs * 1e9)

    algo_bw_pct = 100.0 * algo_gbs / peak_gbs
    compute_pct = 100.0 * achieved_tflops / peak_tflops
    verification_steps: list[str] = []

    if compute_pct >= 85.0:
        hypothesis = (
            f"CANDIDATE: COMPUTE-PIPE NEAR CEILING ({compute_pct:.1f}% of {args.dtype} ceiling). "
            "Further >10% gains likely require algorithmic FLOP reduction or upstream/downstream subgraph fusion."
        )
        verification_steps.append(
            "Confirm pipe saturation via `ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,"
            "smsp__warps_eligible.avg.per_cycle_active` and benchmark against matched cuBLASLt/CUTLASS shapes."
        )
    elif algo_bw_pct >= 85.0:
        hypothesis = (
            f"CANDIDATE: ALGORITHMIC BANDWIDTH NEAR DRAM CEILING ({algo_bw_pct:.1f}% of {peak_gbs} GB/s; "
            ">100% indicates L2 cache hits). Inner-loop compute tweaks will not move wall time; fuse DRAM round-trips."
        )
        verification_steps.append(
            "Note: `bytes / time` measures algorithmic-minimum traffic, not physical DRAM bytes. Confirm actual DRAM "
            "saturation vs L2 hit rate via `ncu --metrics dram__bytes.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,"
            "lts__t_sector_hit_rate.pct` (or run a half-bytes vs half-math control ablation if `ncu` is unavailable)."
        )
    elif ai < ridge_ai:
        if args.time_us < 8.0:
            hypothesis = "CANDIDATE: LAUNCH / TAIL-LATENCY BOUND (<8 us low-AI micro-kernel). Evaluate CUDA Graphs, fusion, or PDL."
            verification_steps.append(
                "Inspect inter-kernel gaps in `nsys` (without `--cuda-graph-trace=node`) and measure window latency after fusion/PDL."
            )
        else:
            hypothesis = (
                f"CANDIDATE: MEMORY-SUBSYSTEM / LATENCY STALLED (Algo AI {ai:.2f} < Ridge {ridge_ai:.2f} FLOP/B, "
                f"algorithmic BW is {algo_bw_pct:.1f}% of DRAM ceiling). Suspect uncoalesced sector overfetch, "
                "L2/DRAM thrashing, or insufficient warp occupancy."
            )
            verification_steps.append(
                "Physical DRAM bandwidth may already be saturated by overfetch. Compare `dram__bytes.sum` against `--bytes`, "
                "and check `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` in `ncu`."
            )
    else:
        hypothesis = (
            f"CANDIDATE: COMPUTE / PIPELINE STALLED (Algo AI {ai:.2f} >= Ridge {ridge_ai:.2f} FLOP/B, "
            f"only {compute_pct:.1f}% of compute ceiling). Suspect memory-pipe stalls (`wait_group 0`), SMEM bank conflicts, "
            "`CALL`/spill overhead, or SKU-specific half-rate MMA."
        )
        verification_steps.append(
            "Run `sass-audit` first, then check `ncu` stall counters (`smsp__warp_issue_stalled_mio_throttle`, "
            "`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`, `dram__bytes.sum.per_second`) and compare with cuBLASLt/CUTLASS."
        )

    report = {
        "gpu": args.gpu,
        "preset_provenance": preset["provenance"] if preset and args.peak_gbs is None and args.peak_tflops is None else "custom user ceiling",
        "dtype": args.dtype,
        "time_us": args.time_us,
        "algorithmic_gbs": round(algo_gbs, 2),
        "peak_dram_gbs": peak_gbs,
        "algorithmic_bw_pct_of_dram": round(algo_bw_pct, 2),
        "achieved_tflops": round(achieved_tflops, 2),
        "peak_tflops": peak_tflops,
        "compute_utilization_pct": round(compute_pct, 2),
        "arithmetic_intensity_flop_per_byte": round(ai, 3),
        "ridge_point_flop_per_byte": round(ridge_ai, 3),
        "hypothesis": hypothesis,
        "verification_steps": verification_steps,
    }
    return emit_output(report, args.output)


def parse_abba_cycles(args: argparse.Namespace) -> list[tuple[float, float, float, float]]:
    if args.cycles:
        cycles = []
        for chunk in args.cycles.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            vals = [float(x.strip()) for x in chunk.split(",")]
            if len(vals) != 4:
                raise ValueError(f"Each ABBA cycle in --cycles must have 4 comma-separated floats (A1,B1,B2,A2); got: {chunk}")
            cycles.append((vals[0], vals[1], vals[2], vals[3]))
        if not cycles:
            raise ValueError("No valid cycles parsed from --cycles.")
        return cycles
    if None in (args.a1, args.b1, args.b2, args.a2):
        raise ValueError("Provide either --a1 --b1 --b2 --a2 OR --cycles 'A1,B1,B2,A2;...'.")
    return [(args.a1, args.b1, args.b2, args.a2)]


def bootstrap_ci_95(values: list[float], n_boot: int = 2000) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(42)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return round(lo, 3), round(hi, 3)


def cmd_abba(args: argparse.Namespace) -> int:
    try:
        cycles = parse_abba_cycles(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cycle_gains: list[float] = []
    cycle_drifts_a: list[float] = []
    cycle_spreads_b: list[float] = []
    means_a: list[float] = []
    means_b: list[float] = []

    for a1, b1, b2, a2 in cycles:
        ma = 0.5 * (a1 + a2)
        mb = 0.5 * (b1 + b2)
        if ma == 0:
            print("Error: mean(A) cannot be zero.", file=sys.stderr)
            return 1
        means_a.append(ma)
        means_b.append(mb)
        gain = 100.0 * (ma - mb) / ma if args.lower_is_better else 100.0 * (mb - ma) / ma
        cycle_gains.append(gain)
        cycle_drifts_a.append(100.0 * abs(a1 - a2) / abs(ma))
        cycle_spreads_b.append(100.0 * abs(b1 - b2) / abs(mb) if mb != 0 else 0.0)

    mean_a = statistics.mean(means_a)
    mean_b = statistics.mean(means_b)
    improvement_pct = statistics.mean(cycle_gains)
    median_improvement_pct = statistics.median(cycle_gains)
    ci95 = bootstrap_ci_95(cycle_gains)
    drift_a_pct = max(cycle_drifts_a)
    spread_b_pct = max(cycle_spreads_b)
    effective_noise_floor_pct = max(args.min_noise_pct, max(drift_a_pct, spread_b_pct) * 0.75)

    flags: list[str] = []
    if drift_a_pct > 2.0:
        flags.append(
            f"THERMAL / CLOCK DRIFT: Max baseline A drift is {drift_a_pct:.2f}%. "
            "Lock power cap (`nvidia-smi -pl`) or SM clocks (`nvidia-smi -lgc`), check `flock`, and increase warm-up."
        )
    if spread_b_pct > 1.0:
        flags.append(
            f"RUN-TO-RUN VARIANCE WARNING: Max candidate B spread is {spread_b_pct:.2f}%. "
            "Check clock throttling, background load, or trajectory-shifting non-determinism (verify determinism separately via output hashes)."
        )
    if ci95 is not None and ci95[0] <= 0.0 <= ci95[1]:
        flags.append(f"STATISTICALLY INDISTINGUISHABLE FROM ZERO: 95% bootstrap CI {ci95} spans 0%.")
    elif len(cycles) == 1 and abs(improvement_pct) < effective_noise_floor_pct:
        flags.append(
            f"NOISE FLOOR: Single-cycle delta ({improvement_pct:+.2f}%) is below effective noise threshold "
            f"({effective_noise_floor_pct:.2f}%); repeat N>=3 ABBA cycles (`--cycles`) or treat as flat."
        )

    spec_decomposition: dict[str, Any] | None = None
    if args.step_ms_a is not None and args.step_ms_b is not None and not args.lower_is_better:
        step_gain_pct = 100.0 * (args.step_ms_a - args.step_ms_b) / args.step_ms_a
        spec_decomposition = {
            "step_ms_a": args.step_ms_a,
            "step_ms_b": args.step_ms_b,
            "hardware_window_speedup_pct": round(step_gain_pct, 3),
        }
        if args.tokens_per_window_a is not None and args.tokens_per_window_b is not None:
            accept_delta_pct = 100.0 * (args.tokens_per_window_b - args.tokens_per_window_a) / args.tokens_per_window_a
            spec_decomposition["tokens_per_window_a"] = args.tokens_per_window_a
            spec_decomposition["tokens_per_window_b"] = args.tokens_per_window_b
            spec_decomposition["accepted_tokens_per_window_delta_pct"] = round(accept_delta_pct, 3)

        if (improvement_pct < 0 and step_gain_pct > 0) or abs(improvement_pct - step_gain_pct) > 2.0:
            flags.append(
                f"SPECULATIVE DRAFT-LUCK CONFOUNDER: tok/s changed by {improvement_pct:+.2f}%, whereas hardware "
                f"window step latency (`ms/window`: {args.step_ms_a} -> {args.step_ms_b} ms) changed by {step_gain_pct:+.2f}%. "
                "Report `ms/window` + `tokens/window` alongside non-speculative `--spec 1`."
            )

    a1, b1, b2, a2 = cycles[0]
    report = {
        "metric": args.metric,
        "lower_is_better": args.lower_is_better,
        "num_cycles": len(cycles),
        "runs": {"A1": a1, "B1": b1, "B2": b2, "A2": a2} if len(cycles) == 1 else [
            {"A1": c[0], "B1": c[1], "B2": c[2], "A2": c[3]} for c in cycles
        ],
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "improvement_pct": round(improvement_pct, 3),
        "median_improvement_pct": round(median_improvement_pct, 3),
        "bootstrap_ci95_pct": list(ci95) if ci95 is not None else None,
        "baseline_drift_pct": round(drift_a_pct, 3),
        "candidate_spread_pct": round(spread_b_pct, 3),
        "effective_noise_floor_pct": round(effective_noise_floor_pct, 3),
        "speculative_decomposition": spec_decomposition,
        "hygiene_flags": flags,
    }
    return emit_output(report, args.output)
