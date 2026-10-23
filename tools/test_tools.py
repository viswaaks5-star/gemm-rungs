"""Tests for tools/. Run: python3 -m unittest discover -s tools -p "test_*.py" -v"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ladder_table  # noqa: E402
import sass_mix  # noqa: E402

LOG = """GIT_SHA: abc1234
device   : NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, 24 SMs, 1536 threads/SM)
software : CUDA runtime 13.1, driver 610.74 (supports CUDA 13.3)

CSV,rung,M,N,K,ms,min_ms,max_ms,gflops,pct_peak,pct_cublas,flop_per_clk_sm,sm_lo,sm_hi,mem_lo,mem_hi,reps_accepted,reps_total,graph
CSV,00_cublas,128,128,128,0.010000,0.009900,0.010100,419.43,3.25,100.00,8.322,2100,2100,8201,8201,7,7,0
CSV,01_naive,128,128,128,0.050000,0.049000,0.051000,83.89,0.65,20.00,1.664,2100,2100,8201,8201,7,7,0
CSV,00_cublas,4096,4096,4096,12.000000,11.900000,12.100000,11453.25,88.80,100.00,227.3,2100,2100,8201,8201,7,7,0
"""

SASS = """
\tcode for sm_89
\t\tFunction : _Z11sgemm_naiveiiifPKfS0_fPf
\t.headerflags\t@"EF_CUDA_SM89"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;   /* 0x00000a00ff017624 */
                                                                             /* 0x000fe400078e00ff */
        /*0010*/                   S2R R0, SR_CTAID.X ;
        /*0020*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x160], PT ;
        /*0030*/               @P0 EXIT ;
        /*0040*/                   LDG.E R2, [R4.64] ;
        /*0050*/                   LDG.E R3, [R6.64] ;
        /*0060*/                   IADD3 R4, P1, R4, 0x4, RZ ;
        /*0070*/                   FFMA R8, R2, R3, R8 ;
        /*0080*/                   LDG.E R2, [R4.64] ;
        /*0090*/                   LDG.E R3, [R6.64+0x4000] ;
        /*00a0*/                   FFMA R8, R2, R3, R8 ;
        /*00b0*/                   ISETP.NE.AND P2, PT, R9, RZ, PT ;
        /*00c0*/               @P2 BRA 0x40 ;
        /*00d0*/                   STG.E [R10.64], R8 ;
        /*00e0*/                   EXIT ;
        /*00f0*/                   BRA 0xf0;
\t\tFunction : _Z11sgemm_tiledv
\t.headerflags\t@"EF_CUDA_SM89"
        /*0000*/                   LDS.U.128 R4, [R12] ;
        /*0010*/                   LDS R8, [R13+0x800] ;
.L_x_1:
        /*0020*/                   FFMA R0, R4, R8, R0 ;
        /*0030*/                   FFMA R1, R5, R8, R1 ;
        /*0040*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
        /*0050*/              @!P0 BRA `(.L_x_1) ;
        /*0060*/                   EXIT ;
"""


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class LadderTable(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.log = os.path.join(self.dir, "2026-10-24_1530_abc1234.txt")
        with open(self.log, "w") as f:
            f.write(LOG)

    def table(self):
        header, rows = ladder_table.parse_log(self.log)
        return ladder_table.make_table(header, rows, "sgemm/runs/2026-10-24_1530_abc1234.txt")

    def test_layout(self):
        t = self.table()
        self.assertIn("| rung | 128³ | 4096³ |", t)
        self.assertIn("| 00_cublas | 419 | 11,453 |", t)
        self.assertIn("| 01_naive | 83.9 (20.0%) | — |", t)   # skipped at 4096: a dash, not a gap
        self.assertIn("2026-10-24 · commit `abc1234`", t)
        self.assertIn("NVIDIA GeForce RTX 4060 Laptop GPU · SM clock 2100 MHz", t)
        self.assertNotIn("uncommitted", t)

    def test_dirty_build_is_flagged(self):
        with open(self.log, "w") as f:
            f.write(LOG.replace("GIT_SHA: abc1234", "GIT_SHA: abc1234-dirty"))
        self.assertIn("Built from uncommitted code", self.table())

    def test_dirty_file_name_is_flagged_too(self):
        header, rows = ladder_table.parse_log(self.log)
        t = ladder_table.make_table(header, rows, "sgemm/runs/2026-10-24_153000_abc1234-dirty.txt")
        self.assertIn("Built from uncommitted code", t)

    def test_readme_splice_is_idempotent(self):
        readme = os.path.join(self.dir, "README.md")
        with open(readme, "w") as f:
            f.write("intro\n<!-- ladder:begin -->\nold\n<!-- ladder:end -->\noutro\n")
        ladder_table.main([self.log, "--readme", readme])
        first = read(readme)
        ladder_table.main([self.log, "--readme", readme])
        self.assertEqual(first, read(readme))
        self.assertNotIn("old", first)
        self.assertTrue(first.startswith("intro\n") and first.endswith("outro\n"))
        self.assertIn("[raw log](2026-10-24_1530_abc1234.txt)", first)

    def test_missing_markers_fail_loudly(self):
        with self.assertRaises(SystemExit):
            ladder_table.splice("no markers here", "table")

    def test_check_only_log_has_no_table(self):
        with open(self.log, "w") as f:
            f.write("GIT_SHA: abc1234\nall configurations verified\n")
        with self.assertRaises(SystemExit):
            self.table()


class SassMix(unittest.TestCase):
    def setUp(self):
        self.funcs = sass_mix.parse(SASS.splitlines(keepends=True))

    def test_functions_found(self):
        self.assertEqual(list(self.funcs), ["_Z11sgemm_naiveiiifPKfS0_fPf", "_Z11sgemm_tiledv"])

    def test_hot_loop_by_address(self):
        insns, _ = self.funcs["_Z11sgemm_naiveiiifPKfS0_fPf"]
        start, end, body = sass_mix.hot_loop(insns)
        self.assertEqual((start, end), (0x40, 0xC0))
        total, ffma, text = sass_mix.summarize(body)
        self.assertEqual((total, ffma), (9, 2))   # 4 LDG, IADD3, 2 FFMA, ISETP, BRA
        self.assertIn("LDG.32 4", text)

    def test_hot_loop_by_label_and_widths(self):
        insns, _ = self.funcs["_Z11sgemm_tiledv"]
        start, end, body = sass_mix.hot_loop(insns)
        self.assertEqual((start, end), (0x20, 0x50))
        self.assertEqual(sass_mix.summarize(body)[:2], (4, 2))
        _, _, whole = sass_mix.summarize(insns)
        self.assertIn("LDS.128 1", whole)
        self.assertIn("LDS.32 1", whole)
        self.assertIn("BAR 1", whole)

    def test_cli_writes_only_matching_sass(self):
        d = tempfile.mkdtemp()
        src, out = os.path.join(d, "all.sass"), os.path.join(d, "naive.sass")
        with open(src, "w") as f:
            f.write(SASS)
        self.assertEqual(sass_mix.main([src, "--match", "sgemm_naive", "--out", out]), 0)
        text = read(out)
        self.assertIn("sgemm_naive", text)
        self.assertNotIn("sgemm_tiled", text)
        self.assertEqual(sass_mix.main([src, "--match", "nonexistent"]), 1)


if __name__ == "__main__":
    unittest.main()
