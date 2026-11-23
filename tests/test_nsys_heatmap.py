"""Unit and integration test suite for nsys-heatmap."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from nsys_heatmap import audit, hierarchy  # noqa: E402


class NsysHeatmapTests(unittest.TestCase):
    def test_discrete_smem_carveout_and_unlaunchable_occupancy(self) -> None:
        bad = audit.compute_occupancy("sm_120a", 128, 32, 101 * 1024)
        self.assertFalse(bad["launchable"])
        self.assertEqual(bad["ctas_per_sm"], 0)
        self.assertIsNone(bad["estimated_best_case_l1_kib"])

        ok_55 = audit.compute_occupancy("sm_120", 256, 128, int(55.8 * 1024))
        self.assertEqual(ok_55["ctas_per_sm"], 1)
        self.assertEqual(ok_55["selected_smem_carveout_kib"], 64)
        self.assertEqual(ok_55["estimated_best_case_l1_kib"], 64.0)

        cliff_74 = audit.compute_occupancy("sm_120", 256, 128, int(74.2 * 1024))
        self.assertEqual(cliff_74["ctas_per_sm"], 1)
        self.assertEqual(cliff_74["selected_smem_carveout_kib"], 100)
        self.assertEqual(cliff_74["estimated_best_case_l1_kib"], 28.0)

        cta_35 = audit.compute_occupancy("sm_120", 256, 77, int(35.8 * 1024))
        self.assertEqual(cta_35["ctas_per_sm"], 2)

        tiny = audit.compute_occupancy("sm_120", 32, 32, 0)
        self.assertEqual(tiny["ctas_per_sm"], 32)
        self.assertEqual(tiny["selected_smem_carveout_kib"], 32)
        self.assertEqual(tiny["estimated_best_case_l1_kib"], 96.0)

    def test_sass_predicates_widths_and_mma(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sass_f = Path(td) / "k.sass"
            res_f = Path(td) / "k.res"
            res_f.write_text("Function _Z4kernv:\n REG:64 STACK:32 SHARED:0\n SPILL_STORES:0 SPILL_LOADS:0\n")
            sass_f.write_text(
                "Function : _Z4kernv\n"
                "/*0000*/ @!UPT LDGSTS.E.BYPASS.LTC128B.128 [R4], [R2.64] ;\n"
                "/*0010*/ @PT HGMMA.64x128x16.F32 R8, desc[UR4], R16 ;\n"
                "/*0020*/ @!P0 STL [R1], R2 ;\n"
            )
            env = {"PYTHONPATH": str(ROOT_DIR / "src")}
            out = subprocess.check_output(
                [sys.executable, "-m", "nsys_heatmap", "sass-audit",
                 "--sass-file", str(sass_f), "--res-file", str(res_f), "--arch", "sm_90", "--threads", "128"],
                text=True, env={**dict(__import__("os").environ), **env},
            )
            data = json.loads(out)["kernels"][0]
            self.assertEqual(data["instruction_summary"]["total_instructions"], 3)
            self.assertEqual(data["instruction_summary"]["wide_LDG_128_256_LDGSTS"], 1)
            self.assertEqual(data["instruction_summary"]["MMA_TensorCore"], 1)
            self.assertEqual(data["instruction_summary"]["MMA_by_family"]["HGMMA"], 1)

    def test_nsys_window_clipping_and_per_gpu_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "t.sqlite"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO StringIds VALUES (1, 'gpu1_gemm')")
            conn.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (shortName INT, demangledName INT, start INT, end INT, "
                "streamId INT, deviceId INT, registersPerThread INT, dynamicSharedMemory INT, staticSharedMemory INT)"
            )
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1, 0, 15000000, 7, 1, 128, 35840, 0)")
            conn.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (copyKind INT, bytes INT, start INT, end INT, deviceId INT)"
            )
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1, 100000000, 5000000, 15000000, 0)")
            conn.commit()
            conn.close()

            env = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT_DIR / "src")}
            out = subprocess.check_output(
                [sys.executable, "-m", "nsys_heatmap", "nsys-top",
                 "--sqlite", str(db), "--limit", "5", "--demangled", "--start-ns", "5000000", "--end-ns", "20000000"],
                text=True, env=env,
            )
            data = json.loads(out)
            self.assertEqual(data["total_kernel_sum_ms"], 10.0)
            self.assertEqual(data["per_device"]["gpu_0"]["memcpy_overlap"]["HtoD"]["overlap_pct"], 0.0)
            self.assertEqual(data["per_device"]["gpu_1"]["merged_compute_active_ms"], 10.0)

    def test_auto_profile_and_multicycle_abba_cli(self) -> None:
        env = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT_DIR / "src")}
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            db_b = tdp / "before.sqlite"
            db_a = tdp / "after.sqlite"
            for idx, dbp in enumerate((db_b, db_a)):
                conn = sqlite3.connect(dbp)
                conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
                conn.execute("INSERT INTO StringIds VALUES (1, 'gemm_kernel_256')")
                conn.execute(
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (shortName INT, demangledName INT, start INT, end INT, "
                    "deviceId INT, gridX INT, gridY INT, gridZ INT, blockX INT, blockY INT, blockZ INT, "
                    "registersPerThread INT, dynamicSharedMemory INT, staticSharedMemory INT)"
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1, 0, 20000000, 0, 110, 1, 1, 256, 1, 1, 128, 65536, 0)"
                )
                conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (copyKind INT, bytes INT, start INT, end INT, deviceId INT)")
                if idx == 0:
                    conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1, 280000000, 20000000, 30000000, 0)")
                else:
                    conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1, 280000000, 2000000, 12000000, 0)")
                conn.commit()
                conn.close()

            prof_json = tdp / "prof.json"
            svg_out = tdp / "out.svg"
            subprocess.check_call(
                [sys.executable, "-m", "nsys_heatmap", "auto",
                 "--mode", "commit-focus", "--left-sqlite", str(db_b), "--right-sqlite", str(db_a),
                 "--profile-out", str(prof_json), "--svg", str(svg_out)],
                env=env, stdout=subprocess.DEVNULL,
            )
            prof = json.loads(prof_json.read_text())
            self.assertIn("host_pcie", prof["focus_targets"])
            self.assertIn("<svg", svg_out.read_text())

            abba_out = subprocess.check_output(
                [sys.executable, "-m", "nsys_heatmap", "abba",
                 "--cycles", "100,110,111,101;100,109,110,100;99,110,110,100"],
                text=True, env=env,
            )
            abba = json.loads(abba_out)
            self.assertEqual(abba["num_cycles"], 3)
            self.assertGreater(abba["bootstrap_ci95_pct"][0], 8.0)


if __name__ == "__main__":
    unittest.main()
