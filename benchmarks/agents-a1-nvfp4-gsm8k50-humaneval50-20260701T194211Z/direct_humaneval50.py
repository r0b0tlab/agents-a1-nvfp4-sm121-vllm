#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:18080/v1'
CHAT = BASE + '/chat/completions'
MODEL = 'agents-a1-nvfp4'
LIMIT = 50
CONCURRENCY = 2

stop_tel = threading.Event()

def telemetry_loop():
    path = OUT / 'telemetry_gpu_direct_humaneval.csv'
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ts','gpu_index','name','util_gpu_pct','temp_c','power_w','graphics_clock_mhz','sm_clock_mhz','mem_clock_mhz','mem_used_mib','mem_total_mib'])
        f.flush()
        while not stop_tel.is_set():
            ts = datetime.now(timezone.utc).isoformat()
            try:
                q = 'index,name,utilization.gpu,temperature.gpu,power.draw,clocks.gr,clocks.sm,clocks.mem,memory.used,memory.total'
                r = subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader,nounits'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5)
                for line in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(',')]
                    parts = ['' if p == '[N/A]' else p for p in parts]
                    if len(parts) >= 10:
                        w.writerow([ts] + parts[:10])
            except Exception as e:
                w.writerow([ts, 'ERR', repr(e), '', '', '', '', '', '', '', ''])
            f.flush()
            stop_tel.wait(2)

def load_tasks():
    from datasets import load_dataset
    ds = load_dataset('openai/openai_humaneval', split='test')
    return [dict(ds[i]) for i in range(min(LIMIT, len(ds)))]

def post(prompt: str):
    payload = {
        'model': MODEL,
        'messages': [
            {'role':'system','content':'You are a Python coding assistant. Return only executable Python code, with no Markdown fences and no prose.'},
            {'role':'user','content': prompt},
        ],
        'temperature': 0,
        'max_tokens': 768,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(CHAT, data=data, headers={'Content-Type':'application/json'})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=240) as r:
        body = json.loads(r.read().decode())
    dt = time.perf_counter() - t0
    msg = body['choices'][0]['message']
    return {'latency_s': dt, 'text': msg.get('content') or msg.get('reasoning') or '', 'raw': body, 'usage': body.get('usage') or {}}

def extract_code(text: str, prompt: str, entry: str):
    # Prefer fenced code if model used Markdown despite instructions.
    m = re.search(r'```(?:python)?\s*(.*?)```', text, flags=re.S | re.I)
    code = m.group(1).strip() if m else text.strip()
    # Strip common lead-in prose.
    code = re.sub(r'^Here is.*?\n', '', code, flags=re.I)
    if re.search(rf'\bdef\s+{re.escape(entry)}\s*\(', code):
        return code
    # Otherwise treat output as completion after the canonical prompt.
    return prompt + code

def evaluate(code: str, test: str, entry: str, timeout=5):
    runner = textwrap.dedent(f'''
    import faulthandler, signal, sys, math, itertools, collections, functools, heapq, bisect, statistics, random, string, re
    faulthandler.enable()
    CODE = {code!r}
    TEST = {test!r}
    ENTRY = {entry!r}
    ns = {{}}
    exec(CODE, ns)
    exec(TEST, ns)
    ns['check'](ns[ENTRY])
    print('PASS')
    ''')
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(runner)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return {'passed': p.returncode == 0, 'returncode': p.returncode, 'stdout': p.stdout[-1000:], 'stderr': p.stderr[-2000:]}
    except subprocess.TimeoutExpired as e:
        return {'passed': False, 'returncode': 'timeout', 'stdout': (e.stdout or '')[-1000:] if isinstance(e.stdout, str) else '', 'stderr': 'TIMEOUT'}
    finally:
        try: os.unlink(path)
        except OSError: pass

def solve_one(task):
    prompt = (
        'Complete the following HumanEval Python function. Return only the complete Python code needed to define the function.\n\n'
        + task['prompt']
    )
    rec = {'task_id': task['task_id'], 'entry_point': task['entry_point'], 'prompt': task['prompt']}
    try:
        resp = post(prompt)
        code = extract_code(resp['text'], task['prompt'], task['entry_point'])
        ev = evaluate(code, task['test'], task['entry_point'])
        rec.update({'ok': True, 'response_text': resp['text'], 'code': code, 'eval': ev, 'usage': resp['usage'], 'latency_s': resp['latency_s'], 'raw_id': resp['raw'].get('id'), 'finish_reason': resp['raw']['choices'][0].get('finish_reason')})
    except Exception as e:
        rec.update({'ok': False, 'error': repr(e)})
    return rec

def summarize_power(path):
    vals = []
    if path.exists():
        for row in csv.DictReader(path.open()):
            try:
                vals.append({k: float(row[k]) for k in ['util_gpu_pct','temp_c','power_w'] if row.get(k) not in ('', None)})
            except Exception:
                pass
    def stat(k):
        xs = [v[k] for v in vals if k in v]
        return {'avg': sum(xs)/len(xs), 'max': max(xs), 'samples': len(xs)} if xs else None
    return {k: stat(k) for k in ['power_w','util_gpu_pct','temp_c']}

def main():
    started = datetime.now(timezone.utc).isoformat()
    tasks = load_tasks()
    tel = threading.Thread(target=telemetry_loop, daemon=True)
    tel.start()
    results = []
    try:
        with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = [ex.submit(solve_one, t) for t in tasks]
            for fut in cf.as_completed(futs):
                rec = fut.result()
                results.append(rec)
                print(json.dumps({'task_id': rec.get('task_id'), 'passed': rec.get('eval',{}).get('passed'), 'latency_s': rec.get('latency_s'), 'completion_tokens': rec.get('usage',{}).get('completion_tokens')}), flush=True)
    finally:
        stop_tel.set(); tel.join(timeout=5)
    results.sort(key=lambda r: r.get('task_id',''))
    passed = sum(1 for r in results if r.get('eval',{}).get('passed'))
    total = len(results)
    summary = {
        'benchmark': 'direct_humaneval_50',
        'model': MODEL,
        'base_url': BASE,
        'limit': LIMIT,
        'concurrency': CONCURRENCY,
        'started_at': started,
        'ended_at': datetime.now(timezone.utc).isoformat(),
        'passed': passed,
        'total': total,
        'pass_at_1': passed / total if total else 0,
        'total_completion_tokens': sum((r.get('usage') or {}).get('completion_tokens') or 0 for r in results),
        'avg_latency_s': sum(r.get('latency_s') or 0 for r in results if r.get('latency_s')) / max(1, sum(1 for r in results if r.get('latency_s'))),
        'telemetry_summary': summarize_power(OUT / 'telemetry_gpu_direct_humaneval.csv'),
    }
    (OUT / 'direct_humaneval50_results.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in results) + '\n')
    (OUT / 'direct_humaneval50_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
