"""Repeat only this small transfer post-processing and compare numerical bytes."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import subprocess
import sys
import tempfile
from analyze_transfer import HERE,sha,read,write

if __name__=='__main__':
    names=('RANKINGS_FROZEN.json','TRANSFER_RESULTS.csv','TRANSFER_SUMMARY.json','SOURCE_AUTHENTICATION.json')
    with tempfile.TemporaryDirectory(prefix='gridium-selector-transfer-',dir='/private/tmp') as name:
        subprocess.run([sys.executable,str(HERE/'analyze_transfer.py'),'--output',name],check=True,
                       stdout=subprocess.DEVNULL,env=os.environ,timeout=70)
        hashes={p:dict(main=sha(HERE/p),repeat=sha(Path(name)/p)) for p in names}
        assert all(v['main']==v['repeat'] for v in hashes.values())
        write(HERE/'REPRODUCIBILITY.json',dict(status='PASS',hashes=hashes,repeat_validation=read(Path(name)/'NUMERICAL_VALIDATION.json')))
    print('PASS: four transfer/provenance outputs reproduced byte for byte')
