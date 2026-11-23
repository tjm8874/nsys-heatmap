"""Unified `nsys-heatmap` CLI entry point."""

from __future__ import annotations

import argparse
import sys

from nsys_heatmap import __version__, audit, hierarchy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nsys-heatmap",
        description="7-Layer CUDA Hardware Efficiency Analyzer, Bottleneck Inference & GitHub PR Heatmap Generator",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1. auto (primary one-shot pipeline)
    p_ap = sub.add_parser(
        "auto",
        aliases=["auto-profile"],
        help="One-shot pipeline: nsys .sqlite + binary/SASS (+ optional ncu CSV) -> JSON + SVG/PNG PR heatmap",
    )
    p_ap.add_argument("--mode", default="commit-focus", choices=["prefill-vs-decode", "commit-focus"])
    p_ap.add_argument("--arch", default="sm_120", help="Target SM architecture (e.g. sm_80, sm_89, sm_90a, sm_100a, sm_120a)")
    p_ap.add_argument("--gpu", default="rtx_pro_5000_300w", choices=sorted(audit.GPU_PRESETS.keys()))
    p_ap.add_argument("--pcie-gbs", type=float, default=28.0, help="PCIe link practical ceiling GB/s (Gen4x16=28, Gen5x16=56)")
    p_ap.add_argument("--device", type=int, help="CUDA deviceId filter (defaults to busiest GPU in trace)")
    p_ap.add_argument("--left-sqlite", required=True, help="Left column nsys .sqlite (Prefill or Before)")
    p_ap.add_argument("--right-sqlite", help="Right column nsys .sqlite (Decode or After; defaults to --left-sqlite)")
    p_ap.add_argument("--left-start-ns", type=int, help="Optional window start (ns) for left trace")
    p_ap.add_argument("--left-end-ns", type=int, help="Optional window end (ns) for left trace")
    p_ap.add_argument("--right-start-ns", type=int, help="Optional window start (ns) for right trace")
    p_ap.add_argument("--right-end-ns", type=int, help="Optional window end (ns) for right trace")
    p_ap.add_argument("--binary", help="Shared compiled binary/.cubin (runs cuobjdump -sass/-res-usage)")
    p_ap.add_argument("--left-binary", help="Override binary for left column (e.g. bin-before)")
    p_ap.add_argument("--right-binary", help="Override binary for right column (e.g. bin-after)")
    p_ap.add_argument("--sass-file", help="Shared pre-dumped cuobjdump -sass text file")
    p_ap.add_argument("--left-sass", help="Override SASS text file for left column")
    p_ap.add_argument("--right-sass", help="Override SASS text file for right column")
    p_ap.add_argument("--res-file", help="Shared pre-dumped cuobjdump -res-usage text file")
    p_ap.add_argument("--left-res", help="Override res-usage file for left column")
    p_ap.add_argument("--right-res", help="Override res-usage file for right column")
    p_ap.add_argument("--left-ncu-csv", help="Optional `ncu --csv` file for left column")
    p_ap.add_argument("--right-ncu-csv", help="Optional `ncu --csv` file for right column")
    p_ap.add_argument("--title", help="Custom chart title")
    p_ap.add_argument("--subtitle", help="Custom chart subtitle")
    p_ap.add_argument("--banner", help="Custom focus banner text")
    p_ap.add_argument("--left-label", help="Custom left column header")
    p_ap.add_argument("--right-label", help="Custom right column header")
    p_ap.add_argument("--focus", help="Comma-separated focus layer keys (auto-detected >=5pt delta if omitted)")
    p_ap.add_argument("--profile-out", help="Save synthesized 7-layer JSON profile for optional manual polish")
    p_ap.add_argument("--report-json", help="Save ranked bottleneck inference JSON")
    p_ap.add_argument("--svg", required=True, help="Output SVG path")
    p_ap.add_argument("--png", help="Optional output PNG path")
    p_ap.add_argument("--width", type=int, default=1920, help="PNG pixel width (default: 1920)")
    p_ap.set_defaults(func=hierarchy.cmd_auto_profile)

    # 2. render
    p_rn = sub.add_parser("render", help="Render 7-layer CUDA hierarchy SVG/PNG scorecard from JSON or preset")
    p_rn.add_argument("--input", help="Path to custom hierarchy profile JSON (omit to use --preset)")
    p_rn.add_argument(
        "--preset", default="commit-focus",
        choices=["commit-focus", "prefill-vs-decode", "basalt-wave3-pr", "basalt-prefill-vs-decode"],
        help="Built-in showcase preset",
    )
    p_rn.add_argument("--focus", help="Override comma-separated focus layer keys (e.g. host_pcie,smem_l1,compute_cores)")
    p_rn.add_argument("--svg", required=True, help="Output SVG path")
    p_rn.add_argument("--png", help="Optional output PNG path (auto-rasterized via rsvg-convert/resvg/sips/inkscape)")
    p_rn.add_argument("--width", type=int, default=1920, help="PNG pixel width (default: 1920)")
    p_rn.add_argument("--report-json", help="Optional output JSON for ranked bottleneck inference")
    p_rn.set_defaults(func=hierarchy.cmd_render)

    # 3. template
    p_dt = sub.add_parser("template", aliases=["dump-template"], help="Export an editable 7-layer JSON profile template")
    p_dt.add_argument(
        "--preset", default="commit-focus",
        choices=["commit-focus", "prefill-vs-decode", "basalt-wave3-pr", "basalt-prefill-vs-decode"],
    )
    p_dt.add_argument("--output", required=True, help="Output template JSON path")
    p_dt.set_defaults(func=hierarchy.cmd_dump_template)

    # 4. nsys-top
    p_nsys = sub.add_parser("nsys-top", help="Summarize Nsight Systems (.sqlite) kernel share & per-GPU PCIe overlap")
    p_nsys.add_argument("--sqlite", required=True, help="Path to nsys exported .sqlite file")
    p_nsys.add_argument("--limit", type=int, default=20, help="Number of top kernels to include (default: 20)")
    p_nsys.add_argument("--demangled", action="store_true", help="Use full demangled kernel signatures")
    p_nsys.add_argument("--device", type=int, help="Filter by CUDA deviceId")
    p_nsys.add_argument("--start-ns", type=int, help="Optional window start timestamp (ns); clips straddling events")
    p_nsys.add_argument("--end-ns", type=int, help="Optional window end timestamp (ns); clips straddling events")
    p_nsys.add_argument("--output", help="Write JSON report to file")
    p_nsys.set_defaults(func=audit.cmd_nsys_top)

    # 5. sass-audit
    p_sass = sub.add_parser("sass-audit", help="Audit SASS instructions, local memory/spills, discrete L1 carveout & occupancy")
    p_sass.add_argument("--binary", help="Path to compiled executable, .o, or .cubin (runs cuobjdump)")
    p_sass.add_argument("--sass-file", help="Path to pre-dumped `cuobjdump -sass` text file")
    p_sass.add_argument("--res-file", help="Path to pre-dumped `cuobjdump -res-usage` text file")
    p_sass.add_argument("--function-filter", help="Substring filter on mangled/demangled function name")
    p_sass.add_argument("--arch", default="sm_120", help="Target SM architecture (e.g. sm_80, sm_89, sm_90a, sm_100a, sm_120a)")
    p_sass.add_argument("--threads", type=int, default=0, help="Threads per CTA (for occupancy & L1 carveout math)")
    p_sass.add_argument("--regs", type=int, help="Override registers per thread if -res-usage is unavailable")
    p_sass.add_argument("--dynamic-smem", type=int, default=0, help="Dynamic shared memory bytes per CTA")
    p_sass.add_argument("--output", help="Write JSON report to file")
    p_sass.set_defaults(func=audit.cmd_sass_audit)

    # 6. sass-metrics
    p_sm = sub.add_parser("sass-metrics", help="Extract Warp SIMT branchless ratio, LDG width score & TC dtype/shape from SASS")
    p_sm.add_argument("--binary", help="Compiled binary/.cubin/.o (invokes cuobjdump -sass)")
    p_sm.add_argument("--sass-file", help="Saved cuobjdump -sass text file")
    p_sm.add_argument("--function-filter", help="Optional function filter for cuobjdump -fun")
    p_sm.add_argument("--output", help="Output JSON file path")
    p_sm.set_defaults(func=hierarchy.cmd_sass_metrics)

    # 7. roofline
    p_roof = sub.add_parser("roofline", help="Compute algorithmic GB/s, TFLOP/s, AI, and candidate bottleneck hypotheses")
    p_roof.add_argument("--gpu", default="rtx_pro_5000_300w", choices=sorted(audit.GPU_PRESETS.keys()))
    p_roof.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16", "int8", "fp8", "mxfp8", "nvfp4"])
    p_roof.add_argument("--bytes", type=float, required=True, help="Minimum algorithmic DRAM bytes moved")
    p_roof.add_argument("--flops", type=float, required=True, help="Total arithmetic FLOPs / OPs executed")
    p_roof.add_argument("--time-us", type=float, required=True, help="Measured kernel runtime in microseconds")
    p_roof.add_argument("--peak-gbs", type=float, help="Custom sustained DRAM bandwidth ceiling (GB/s)")
    p_roof.add_argument("--peak-tflops", type=float, help="Custom sustained compute ceiling (TFLOP/s or TOP/s)")
    p_roof.add_argument("--output", help="Write JSON report to file")
    p_roof.set_defaults(func=audit.cmd_roofline)

    # 8. abba
    p_abba = sub.add_parser("abba", help="Validate 1 or N interleaved A1 -> B1 -> B2 -> A2 benchmark cycles")
    p_abba.add_argument("--metric", default="throughput_tok_s", help="Metric label")
    p_abba.add_argument("--a1", type=float, help="Single-cycle A1")
    p_abba.add_argument("--b1", type=float, help="Single-cycle B1")
    p_abba.add_argument("--b2", type=float, help="Single-cycle B2")
    p_abba.add_argument("--a2", type=float, help="Single-cycle A2")
    p_abba.add_argument("--cycles", help="Semicolon-separated N cycles: 'A1,B1,B2,A2;A1,B1,B2,A2;...'")
    p_abba.add_argument("--min-noise-pct", type=float, default=0.5, help="Minimum single-ABBA noise floor %% (default: 0.5)")
    p_abba.add_argument("--lower-is-better", action="store_true", help="Set when metric is latency (ms/us)")
    p_abba.add_argument("--step-ms-a", type=float, help="Optional mean ms/window for A (speculative decode check)")
    p_abba.add_argument("--step-ms-b", type=float, help="Optional mean ms/window for B (speculative decode check)")
    p_abba.add_argument("--tokens-per-window-a", type=float, help="Optional mean accepted+bonus tokens/window for A")
    p_abba.add_argument("--tokens-per-window-b", type=float, help="Optional mean accepted+bonus tokens/window for B")
    p_abba.add_argument("--output", help="Write JSON report to file")
    p_abba.set_defaults(func=audit.cmd_abba)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
