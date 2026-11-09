"""Local wiring checks using only explicitly created scheduler/runtime stubs."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='demo paths ')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.project = self.base / 'project copy'
        shutil.copytree(ROOT / 'jobs', self.project / 'jobs')
        self.bin = self.base / 'fake commands'
        self.bin.mkdir()
        self.calls = self.base / 'calls.jsonl'
        self.runtime_calls = self.base / 'runtime.json'
        self.stub('sbatch', '''import json, os, sys
from pathlib import Path
p = Path(os.environ['CALLS'])
n = len(p.read_text().splitlines()) if p.exists() else 0
with p.open('a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')
if os.environ.get('FAIL_SECOND') and n == 1: sys.exit(1)
if os.environ.get('MALFORMED') and n == 1: print('uncertain'); sys.exit(0)
print(str(101 + n) + ';test-cluster')
''')
        self.stub('runtime', '''import json, os, sys
from pathlib import Path
Path(os.environ['RUNTIME_CALLS']).write_text(json.dumps(sys.argv[1:]))
''')
        self.profile = self.base / 'private profile.sh'
        values = dict(PROJECT=self.project, RUN_ROOT=self.base / 'private runs',
                      RUNTIME=self.bin / 'runtime', IMAGE=self.base / 'image file.sif')
        self.profile.write_text(''.join(f'{key}={shlex.quote(str(value))}\n'
                                       for key, value in values.items()) +
                                'ARRAY_CONCURRENCY=2\nPREPARE_SBATCH_OPTIONS=("--cpus-per-task=1")\n'
                                'TRAIN_SBATCH_OPTIONS=()\nAGGREGATE_SBATCH_OPTIONS=()\n')
        self.env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'],
                        CALLS=str(self.calls), RUNTIME_CALLS=str(self.runtime_calls))
        # Fail rather than ever falling back to a system scheduler.
        self.assertEqual(Path(shutil.which('sbatch', path=self.env['PATH'])).resolve(),
                         (self.bin / 'sbatch').resolve())

    def stub(self, name, body):
        path = self.bin / name
        path.write_text('#!/usr/bin/env python3\n' + body)
        path.chmod(0o700)

    def submit(self, *args, **env):
        return subprocess.run(['bash', str(self.project / 'jobs/submit.sh'),
                               str(self.profile), *args], env=dict(self.env, **env),
                              capture_output=True, text=True)

    def test_chain_and_spooled_mapping(self):
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        self.assertIn('--array=0-2%2', calls[1])
        self.assertIn('--dependency=afterok:101', calls[1])
        self.assertIn('--dependency=afterok:102', calls[2])
        self.assertTrue(any('train-%A_%a.out' in arg for arg in calls[1]))
        spooled = self.base / 'spooled batch'
        shutil.copy(self.project / 'jobs/job.sbatch', spooled)
        for stage, call in zip(('prepare', 'train', 'aggregate'), calls):
            index = call.index(str(self.project / 'jobs/job.sbatch'))
            for fold in ((0, 1, 2) if stage == 'train' else (0,)):
                proc = subprocess.run(['bash', str(spooled), *call[index + 1:]],
                                      env=dict(self.env, SLURM_ARRAY_TASK_ID=str(fold)),
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                runtime = json.loads(self.runtime_calls.read_text())
                self.assertIn(f'{self.project}:/project:ro', runtime)
                if stage == 'train':
                    self.assertIn(f'/run/data/fold-{fold}/train.npy', runtime)
                    self.assertIn(f'/run/results/fold-{fold}.json', runtime)
                    self.assertNotIn('--fold', runtime)
        run = Path(result.stdout.splitlines()[0].removeprefix('Run: '))
        self.assertEqual((run.stat().st_mode & 0o777), 0o700)
        self.assertEqual((run / 'receipts/train.stdout').read_text(), '102;test-cluster\n')

    def test_partial_failure_and_malformed_response(self):
        for key in ('FAIL_SECOND', 'MALFORMED'):
            with self.subTest(key=key):
                self.calls.unlink(missing_ok=True)
                result = self.submit(**{key: '1'})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(self.calls.read_text().splitlines()), 2)
                self.assertIn('prepare=101;test-cluster', result.stderr)
                self.assertIn('do not retry blindly', result.stderr)

    def test_preview_does_not_resolve_scheduler(self):
        (self.bin / 'sbatch').unlink()
        result = self.submit('--preview')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertIn('afterok:', result.stdout)
        self.assertIn('prepare-jobid', result.stdout)

    def test_invalid_fold_rejected(self):
        proc = subprocess.run(['bash', str(self.project / 'jobs/run.sh'), 'train',
                               str(self.project), str(self.base), str(self.profile), '3'],
                              env=self.env, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(self.runtime_calls.exists())

    def test_shell_syntax(self):
        for file in (*ROOT.joinpath('jobs').iterdir(), ROOT / 'config.example.sh'):
            subprocess.run(['bash', '-n', str(file)], check=True)


if __name__ == '__main__':
    unittest.main()
