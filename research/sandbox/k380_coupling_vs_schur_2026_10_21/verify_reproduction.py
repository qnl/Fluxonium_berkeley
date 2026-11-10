"""Second small saved-data pass; never calls completed fingerprint analysis."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import subprocess
import sys
import tempfile
from analyze import HERE,sha,read,write

if __name__=='__main__':
    files=('SHELL_COMPARISON.csv','RANKING_ANALYSIS.csv','TARGET_DIAGNOSTICS.csv',
           'CONTEXT_RANK328.json','RESULTS.json','SOURCE_AUTHENTICATION.json')
    with tempfile.TemporaryDirectory(prefix='gridium-coupling-schur-',dir='/private/tmp') as name:
        subprocess.run([sys.executable,str(HERE/'analyze.py'),'--output',name],env=os.environ,
                       check=True,stdout=subprocess.DEVNULL,timeout=190)
        hashes={f:dict(main=sha(HERE/f),repeat=sha(Path(name)/f)) for f in files}
        assert all(h['main']==h['repeat'] for h in hashes.values())
        write(HERE/'REPRODUCIBILITY.json',dict(status='PASS',hashes=hashes,
              repeated_validation=read(Path(name)/'RUN_VALIDATION.json')))
    print('PASS: six scientific/provenance outputs reproduced byte for byte')
