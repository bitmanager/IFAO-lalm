#!/usr/bin/env python3
"""Read native AGY /usage over SSH; does not submit inference requests.

Requires pexpect on the operator host and initialized, running account containers.
Run: python3 infra/annotation/check_usage.py --output /tmp/agy-usage.json
Add --publish to update the remote S3 usage snapshot with these observations.
"""
import pexpect,concurrent.futures,re,json,time,argparse,subprocess,base64,inspect,sys
from datetime import datetime, timezone
from pathlib import Path
import shlex

def merge_results(existing, measurements):
 """Replace one complete observation; never freshen an older raw capture."""
 fields=('usage','weekly_pct','weekly_reset','five_hour_pct','five_hour_reset',
         'cleanup_confirmed','cleanup_method','paid_tier_checked_at','paid_tier_id',
         'paid_tier_name','paid_tier_http_status','paid_tier_error','error','stage')
 rows=[dict(row) for row in existing]
 by_id={row['id']:row for row in rows}
 if len(by_id)!=len(rows):raise ValueError('duplicate_usage_account')
 for observation in measurements:
  account=observation['account'];stamp=observation['checked_at']
  measured=datetime.fromisoformat(stamp)
  if measured.tzinfo is None:raise ValueError('usage_timestamp_requires_timezone')
  row=by_id.get(account)
  if row is not None and row.get('at') and datetime.fromisoformat(row['at'])>measured:continue
  if row is None:
   row={'id':account};rows.append(row);by_id[account]=row
  for field in fields:row.pop(field,None)
  row.update({field:observation[field] for field in fields if field in observation})
  row.update(at=stamp,quota_observation_kind='stock_cli_usage_readonly',
             quota_observed_via='agy /usage; loadCodeAssist paidTier')
 return rows

def publish_results(results):
 # Execute only the merge and S3 publication in the existing scheduler container.
 # Shell quoting and base64 keep captured text out of executable shell syntax.
 payload=base64.b64encode(json.dumps(results,ensure_ascii=False).encode()).decode()
 code=('import base64,json,hashlib\nfrom datetime import datetime\n'
       +inspect.getsource(merge_results)+'''
from lalm.annotation.store import s3,dataset_bucket
client=s3();bucket=dataset_bucket()
response=client.get_object(Bucket=bucket,Key='analytics/usage/latest.json')
try:existing=json.loads(response['Body'].read())
finally:response['Body'].close()
fresh=json.loads(base64.b64decode('''+repr(payload)+'''))
merged=merge_results(existing,fresh)
body=json.dumps(merged,ensure_ascii=False,sort_keys=True).encode()
key='analytics/usage/'+hashlib.sha256(body).hexdigest()+'.json'
client.put_object(Bucket=bucket,Key=key,Body=body)
client.put_object(Bucket=bucket,Key='analytics/usage/latest.json',Body=body,IfMatch=response['ETag'])
print(json.dumps({'immutable_key':key,'measured_rows':len(fresh),'total_rows':len(merged)}))
''')
 command=(f'cd {shlex.quote(ARGS.remote_dir)} && sudo -n infra/annotation/dc exec -T '
          f'airflow-scheduler python -c {shlex.quote(code)}')
 try:
  result=subprocess.run([ARGS.ssh,ARGS.host,command],capture_output=True,text=True,timeout=60)
  if result.returncode:raise RuntimeError('usage_publish_failed')
  return json.loads(result.stdout)
 except (subprocess.TimeoutExpired,OSError,ValueError):
  raise RuntimeError('usage_publish_unavailable') from None

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
 parser.add_argument('--publish',action='store_true',
                     help='Publish measurements to the existing remote S3 usage snapshot')
 ARGS=parser.parse_args()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  results=list(pool.map(check,ARGS.accounts))
 payload=json.dumps(results,ensure_ascii=False,indent=2)+'\n'
 if ARGS.output:ARGS.output.write_text(payload)
 print(payload,end='')
 if ARGS.publish:
  try:print(json.dumps({'publication':publish_results(results)}),file=sys.stderr)
  except RuntimeError as error:
   print(json.dumps({'publication_error':str(error)}),file=sys.stderr)
   raise SystemExit(1)
 raise SystemExit(1 if any('error' in r for r in results) else 0)
