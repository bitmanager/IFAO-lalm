#!/usr/bin/env python3
"""Read native AGY /usage over SSH; does not submit inference requests.

Requires pexpect on the operator host and initialized, running account containers.
Run: python3 infra/annotation/check_usage.py --output /tmp/agy-usage.json
"""
import pexpect,concurrent.futures,re,json,time,argparse,subprocess
from datetime import datetime, timezone
from pathlib import Path
import shlex

def clean(s):
 s=re.sub(r'\x1b\][^\x07]*(?:\x07)', '', s)
 s=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '',s)
 s=re.sub(r'[\w.+-]+@[\w.-]+','[account]',s)
 return s

def quota_fields(capture):
 """The later Claude/GPT section must never replace Gemini's empty quota."""
 section=capture.split('CLAUDE AND GPT MODELS',1)[0]
 result={}
 for name,heading in [('weekly','Weekly'),('five_hour','Five Hour')]:
  match=re.search(heading+r' Limit Remaining\s*\[[^\]]*\]\s*([0-9.]+)%\s*'
                  r'(?:Refreshes in ([^\r\n]+)|Quota available)',section)
  if match:
   result[name+'_pct']=float(match.group(1))
   result[name+'_reset']=match.group(2) or 'available'
 return result

def check_plan(account):
 # Only a read-only discovery call: no audio or generateContent request.
 # Never print the OAuth file, full provider response, or subprocess errors.
 code="""import json
from lalm.annotation.native_api import post,token
try:
 status,raw=post('loadCodeAssist',{'metadata':{'ideType':'ANTIGRAVITY'}},token(),35)
 data=json.loads(raw) if status==200 else {}
 tier=data.get('paidTier') or {}
 print(json.dumps({'http_status':status,'id':tier.get('id'),'name':tier.get('name')}))
except Exception as error:
 print(json.dumps({'error_type':type(error).__name__}))
"""
 cmd=(f'cd {shlex.quote(ARGS.remote_dir)} && sudo -n infra/annotation/dc exec -T '
      f'{shlex.quote(account)} python -c {shlex.quote(code)}')
 try:
  process=subprocess.run([ARGS.ssh,ARGS.host,cmd],capture_output=True,text=True,timeout=50)
  if process.returncode:
   return {'paid_tier_error':'probe_process_failed'}
  value=json.loads(process.stdout)
  result={'paid_tier_checked_at':datetime.now(timezone.utc).isoformat()}
  if value.get('http_status')==200 and isinstance(value.get('id'),str):
   result.update(paid_tier_id=value['id'],paid_tier_name=value.get('name'))
  else:
   result['paid_tier_error']='discovery_failed'
  if isinstance(value.get('http_status'),int):
   result['paid_tier_http_status']=value['http_status']
  return result
 except (subprocess.TimeoutExpired,OSError,ValueError,TypeError,AttributeError):
  return {'paid_tier_error':'probe_unavailable'}

def check(n):
 account=f'agy-{n:02d}'
 # Bound the process inside Docker too: closing SSH alone can leave its TUI alive.
 cmd=f'cd {shlex.quote(ARGS.remote_dir)} && sudo -n infra/annotation/dc exec {account} sh -c "stty rows 60 cols 150; exec timeout --foreground --kill-after=5s 180s agy"'
 p=pexpect.spawn(ARGS.ssh,['-tt',ARGS.host,cmd],encoding='utf-8',timeout=70,dimensions=(60,150))
 stage='startup'
 try:
  k=p.expect(['Choose your color scheme','Terms of Service & Data Use','Do you trust','for shortcuts','is not running',pexpect.EOF])
  if k != 3:
   stage='interactive_setup' if k<3 else 'container_unavailable' if k==4 else 'process_eof'
   raise RuntimeError('Interactive setup required; complete it manually before polling usage')
  p.send('/usage\r')
  stage='usage'
  p.expect('Models & Quota')
  matched=p.expect('Within each group|Failed to|Error:',timeout=70)
  if matched != 0: raise RuntimeError('Usage endpoint returned an error')
  capture=clean(p.before+p.after)
  result={'account':account,'checked_at':datetime.now(timezone.utc).isoformat(),'usage':capture,
          **quota_fields(capture)}
  p.send('\x1b');time.sleep(.3);p.send('/exit\r')
  try:
   p.expect(pexpect.EOF,timeout=10)
   result['cleanup_confirmed']=True
   result['cleanup_method']='exit_command'
  except pexpect.TIMEOUT:
   # This PTY runs only the interactive /usage reader, never a print-mode job.
   # A modal can swallow /exit; the CLI's double interrupt exits that TUI.
   result['cleanup_method']='double_interrupt'
   try:
    p.sendcontrol('c');time.sleep(.3)
    if p.isalive():p.sendcontrol('c')
    p.expect(pexpect.EOF,timeout=5)
    result['cleanup_confirmed']=True
   except (pexpect.TIMEOUT,pexpect.EOF,OSError):
    result['cleanup_confirmed']=not p.isalive()
 except Exception as e:
  result={'account':account,'checked_at':datetime.now(timezone.utc).isoformat(),'error':type(e).__name__,'stage':stage}
 finally:
  p.close(force=True)
 if 'usage' in result:
  result.update(check_plan(account))
 return result
if __name__ == '__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--ssh',default=str(Path.home()/'.local/bin/ssh-gpu'))
 parser.add_argument('--host',default='dev-1')
 parser.add_argument('--remote-dir',default='/mnt/local/drive2/speech-annotation-20261012/code')
 parser.add_argument('--accounts',nargs='+',type=int,choices=range(1,26),default=list(range(1,13)))
 parser.add_argument('--output',type=Path)
 ARGS=parser.parse_args()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  results=list(pool.map(check,ARGS.accounts))
 payload=json.dumps(results,ensure_ascii=False,indent=2)+'\n'
 if ARGS.output:ARGS.output.write_text(payload)
 print(payload,end='')
 raise SystemExit(1 if any('error' in r for r in results) else 0)
