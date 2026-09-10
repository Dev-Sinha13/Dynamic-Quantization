"""Atomic checkpoints and cooperative wall-time limits for notebook phases."""
import hashlib
import json
import time
from pathlib import Path


class RunControl:
    def __init__(self, directory, manifest, clock=time.perf_counter):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        path = self.directory / 'resume-manifest.json'
        if path.exists() and json.loads(path.read_text())['sha256'] != digest:
            raise ValueError('Checkpoint configuration/source/environment mismatch. Use a new output directory.')
        self.save('resume-manifest.json', {'sha256': digest, 'manifest': manifest})

    def load(self, filename):
        path = self.directory / filename
        return json.loads(path.read_text()) if path.exists() else []

    def save(self, filename, value):
        path = self.directory / filename
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    def start(self, phase, minutes):
        if minutes <= 0:
            raise ValueError('Phase time limit must be positive')
        self.phase = phase
        self.started = self.clock()
        self.deadline = self.started + minutes * 60
        self.paused = False

    def allow(self):
        if self.clock() < self.deadline:
            return True
        if not self.paused:
            print(f'{self.phase}: time budget reached. Saved work retained; rerun to continue.', flush=True)
            self.save(f'{self.phase}-status.json', {'status': 'paused_time_budget'})
        self.paused = True
        return False

    def log(self, message):
        elapsed = (self.clock() - self.started) / 60
        print(f'[{self.phase} {elapsed:.1f} min] {message}', flush=True)
