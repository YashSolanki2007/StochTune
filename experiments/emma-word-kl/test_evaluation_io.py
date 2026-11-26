"""Regress the real Unicode JSONL failure and checkpoint-before-evaluation path."""

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from common import REPO, ROOT, read_rows, write_rows


class EvaluationIOTests(unittest.TestCase):
    def test_generated_unicode_separators_remain_inside_records(self):
        rows = [{"id": "one", "passage": "before\u2028after\u2029paragraph\x85next"},
                {"id": "two", "passage": "literal\nnewline and café"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            write_rows(path, rows)
            # Reproduces the failure mode of the previous shared reader.
            with self.assertRaises(json.JSONDecodeError):
                [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(read_rows(path), rows)
            self.assertEqual(path.read_bytes().count(b"\n"), 2)

    def test_crlf_and_blank_lines_work_but_invalid_json_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_bytes(b'\r\n{"id": 1}\r\n  \r\n{"id": 2}\r\n')
            self.assertEqual(read_rows(path), [{"id": 1}, {"id": 2}])
            path.write_bytes(b'{"id": 1}\n{"passage": "unfinished\n')
            with self.assertRaises(json.JSONDecodeError):
                read_rows(path)

    def test_actual_final_step_checkpoint_survives_evaluation_exception(self):
        # Execute the actual train-loop checkpoint/validation blocks with CPU
        # doubles, rather than reproducing their order in a second implementation.
        implementation = ROOT / "_policy_impl.py"
        if not implementation.exists():
            implementation = REPO / "experiments/helpsteer-rl-v2/train_policy.py"
        tree = ast.parse(implementation.read_text())
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        loop = next(n for n in ast.walk(train) if isinstance(n, ast.While))
        blocks = [n for n in loop.body if isinstance(n, ast.If) and
                  any(key in ast.unparse(n.test) for key in ["checkpoint_every", "validation_every"])]
        self.assertEqual(len(blocks), 2)
        module = ast.fix_missing_locations(ast.Module(body=blocks, type_ignores=[]))
        saved = []

        def save(*args):
            saved.append(args[5])

        def fail(*args):
            raise RuntimeError("evaluation interrupted")

        cfg = {"steps": 500, "stop_after_steps": 100, "checkpoint_every": 30, "validation_every": 20}
        scope = dict.fromkeys(["policy", "tokenizer", "optimizer", "scheduler", "prompt_rng",
                              "output", "device", "reward", "target"], None)
        scope.update(cfg=cfg, step=100, step_limit=100, hashlib=hashlib, json=json, arm="quality-unigram",
                     cases={"validation": []}, save_checkpoint=save, evaluate=fail)
        with self.assertRaisesRegex(RuntimeError, "evaluation interrupted"):
            exec(compile(module, "actual-train-checkpoint-blocks", "exec"), scope)
        self.assertEqual(saved, [100])


if __name__ == "__main__":
    unittest.main()
