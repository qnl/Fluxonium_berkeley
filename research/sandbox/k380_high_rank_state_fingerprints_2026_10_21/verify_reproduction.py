"""Run a second saved-data pass and byte-compare deterministic numerical outputs."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import subprocess
import sys
import tempfile
from pathlib import Path
from analyze import HERE, sha, write, read

if __name__=='__main__':
    files=['FINGERPRINTS.csv','OPERATOR_COUPLINGS.csv','OPERATOR_BLOCKS.npz',
           'DIAGNOSTICS.json','SOURCE_AUTHENTICATION.json']
    with tempfile.TemporaryDirectory(prefix='gridium-fingerprint-',dir='/private/tmp') as name:
        subprocess.run([sys.executable,str(HERE/'analyze.py'),'--output',name],check=True,env=os.environ,
                       stdout=subprocess.DEVNULL,timeout=190)
        hashes={f:dict(first=sha(HERE/f),repeat=sha(Path(name)/f)) for f in files}
        assert all(h['first']==h['repeat'] for h in hashes.values())
        write(HERE/'REPRODUCIBILITY.json',dict(status='PASS',hashes=hashes,
              repeated_run=read(Path(name)/'RUN_VALIDATION.json'),new_eigensolves=0))
    print('PASS: all five numerical artifacts reproduced byte for byte')
