#!/usr/bin/env python3
"""Read native AGY /usage over SSH; does not submit inference requests.

Requires pexpect on the operator host and initialized, running account containers.
Run: python3 infra/annotation/check_usage.py --output /tmp/agy-usage.json
"""
import pexpect,concurrent.futures,re,json,time,argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex

def clean(s):
 s=re.sub(r'\x1b\][^\x07]*(?:\x07)', '', s)
 s=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '',s)
 s=re.sub(r'[\w.+-]+@[\w.-]+','[account]',s)
 return s

def check(n):
 account=f'agy-{n:02d}'
 cmd=f'cd {shlex.quote(ARGS.remote_dir)} && infra/annotation/dc exec {account} sh -c "stty rows 60 cols 150; exec agy"'
 p=pexpect.spawn(ARGS.ssh,['-tt',ARGS.host,cmd],encoding='utf-8',timeout=70,dimensions=(60,150))
 try:
  k=p.expect(['Choose your color scheme','Terms of Service & Data Use','Do you trust','for shortcuts','is not running',pexpect.EOF])
  if k != 3:
   raise RuntimeError('Interactive setup required; complete it manually before polling usage')
  p.send('/usage\r')
  p.expect('Models & Quota')
  matched=p.expect('Within each group|Failed to|Error:',timeout=70)
  if matched != 0: raise RuntimeError('Usage endpoint returned an error')
  capture=clean(p.before+p.after)
  result={'account':account,'checked_at':datetime.now(timezone.utc).isoformat(),'usage':capture}
  p.send('\x1b');time.sleep(.3);p.send('/exit\r')
 except Exception as e:
  result={'account':account,'checked_at':datetime.now(timezone.utc).isoformat(),'error':type(e).__name__}
 finally:
  p.close(force=True)
 return result
if __name__ == '__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--ssh',default=str(Path.home()/'.local/bin/ssh-gpu'))
 parser.add_argument('--host',default='dev-1')
 parser.add_argument('--remote-dir',default='/mnt/local/drive2/speech-annotation-20261012/code')
 parser.add_argument('--accounts',nargs='+',type=int,choices=range(1,17),default=list(range(1,12)))
 parser.add_argument('--output',type=Path)
 ARGS=parser.parse_args()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  results=list(pool.map(check,ARGS.accounts))
 payload=json.dumps(results,ensure_ascii=False,indent=2)+'\n'
 if ARGS.output:ARGS.output.write_text(payload)
 print(payload,end='')
 raise SystemExit(1 if any('error' in r for r in results) else 0)
