"""Audit and ZIP the compact repository, excluding local runs and caches."""
from pathlib import Path
import hashlib
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT/'tools/audit_release.py')], check=True, cwd=ROOT)
    files = [path for path in ROOT.rglob('*') if path.is_file()
             and not any(part in {'.git', '.venv', '__pycache__', 'generated'} for part in path.relative_to(ROOT).parts)]
    sums = ROOT/'provenance/SHA256SUMS'
    files = sorted(path for path in files if path != sums)
    sums.write_text(''.join(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.relative_to(ROOT).as_posix()+'\n'
                            for path in files))
    target = ROOT.parent/'winter-rwis-simulation-github.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files+[sums]:
            archive.write(path, Path(ROOT.name)/path.relative_to(ROOT))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    print(f'ZIP ready: {target.name}; {target.stat().st_size/1024**2:.2f} MiB; {len(files)+1} files')


if __name__ == '__main__':
    main()
