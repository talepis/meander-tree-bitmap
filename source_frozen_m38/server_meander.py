#!/usr/bin/env python3
"""Sequential exact family jobs with fresh RAM guards and completed-job resume."""
import argparse,json,math,os,subprocess,sys,threading,time,traceback
from pathlib import Path
from common import HERE,GiB,Resources,JobLock,active_numeric_jobs,aggregate,choose_threads,planning,save,sha,source_digest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('n',type=int,nargs='?',default=38)
    parser.add_argument('--groups',type=int,choices=(1,2,3,4),default=4)
    parser.add_argument('--threads',default='auto')
    parser.add_argument('--backend',choices=('32','64'),default='64',help='32 is an unchanged unified reference for smaller-order comparisons')
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--memory-gib',type=float,help='Optional additional limit; never raises a cgroup/host allocation')
    parser.add_argument('--reserve-gib',type=float,default=16)
    parser.add_argument('--timeout-hours',type=float,default=24)
    parser.add_argument('--warmup',type=int,choices=(12,),help='Diagnostic warm-up per clean worker; omitted in the M38 launcher')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 1<=args.n<=38:parser.error('Order must be 1..38')
    if not math.isfinite(args.reserve_gib)or args.reserve_gib<3:parser.error('Reserve must be finite and at least 3 GiB')
    if not math.isfinite(args.timeout_hours)or args.timeout_hours<=0:parser.error('Invalid timeout')
    if args.memory_gib is not None and(not math.isfinite(args.memory_gib)or args.memory_gib<=0):parser.error('Invalid memory limit')
    if args.backend=='32'and(args.groups!=1 or args.n>37):parser.error('The reference backend accepts one unified group, n<=37')
    resources=Resources(args.memory_gib);threads=choose_threads(args.threads,resources)
    report=planning(args.n,args.groups,args.backend,resources,args.reserve_gib,threads)
    if args.inspect:
        print(json.dumps(report,indent=2));return 0
    if report['initial_admission']!='READY_FOR_GUARDED_GROUPS':raise MemoryError('Fixed arrays and reserve do not fit; no numerical job started')
    from verify_bundle import main as verify
    verify()
    receipt_path=HERE/'SELF_TEST_RESULT.json'
    if not receipt_path.is_file():raise RuntimeError('Run self_test.py before a numerical job')
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('status')!='PASS'or receipt.get('source_digest')!=report['source_digest']:
        raise RuntimeError('Passing self-test for these exact sources is required')
    if args.n==38 and receipt.get('rank64_boundary_status')!='PASS':raise RuntimeError('M38 requires the large-rank boundary tests')
    existing=active_numeric_jobs()
    if existing:raise RuntimeError('Another numerical MEANDER job is active: '+str(existing))
    lock=JobLock(HERE/'.NUMERICAL_SERIES.lock')
    output=(args.output or HERE/'runs'/('M%d_G%d'%(args.n,len(report['groups'])))).resolve()
    process=None;start=time.monotonic()
    config=dict(n=args.n,groups=report['groups'],backend=args.backend,threads=threads,reserve_bytes=report['reserve_bytes'],
        memory_gib=args.memory_gib,warmup_n=args.warmup,source_digest=report['source_digest'],matrix_files_allowed=False)
    records=[]
    try:
        if output.exists()and any(output.iterdir())and not args.resume:raise FileExistsError('Choose a fresh output or --resume')
        output.mkdir(parents=True,exist_ok=True)
        previous=output/'CONFIG.json'
        if previous.is_file()and json.loads(previous.read_text())!=config:raise RuntimeError('Existing run configuration/source differs; use a new output directory')
        save(previous,config);save(output/'PREFLIGHT.json',report)
        if (output/'RESULT.json').is_file():
            for i,g in enumerate(report['groups'],1):
                p=output/('group_%02d'%i)/'RESULT.json';r=json.loads(p.read_text())
                state=json.loads(p.with_name('STATE.json').read_text())
                if state.get('result_sha256')!=sha(p):raise ArithmeticError('Completed result hash mismatch')
                records.append(r)
            value,covered=aggregate(args.n,records,report['groups'],report['source_digest'])
            final=json.loads((output/'RESULT.json').read_text())
            if final.get('status')!='PASS'or final.get('Mn')!=str(value):raise ArithmeticError('Existing aggregate mismatch')
            print(json.dumps(dict(status='ALREADY_COMPLETE',n=args.n,Mn=str(value),output=str(output))));return 0
        for i,group in enumerate(report['groups'],1):
            groupdir=output/('group_%02d'%i);result_path=groupdir/'RESULT.json'
            if result_path.is_file():
                r=json.loads(result_path.read_text());state=json.loads((groupdir/'STATE.json').read_text())
                if state.get('status')!='PASS'or state.get('result_sha256')!=sha(result_path):raise ArithmeticError('Group result integrity failed')
                if r.get('lower_sizes')!=group or r.get('source_digest')!=report['source_digest']or r.get('n')!=args.n:raise ArithmeticError('Resume provenance mismatch')
                records.append(r);print('Keeping completed group %d/%d'%(i,len(report['groups'])),flush=True);continue
            deadline=args.timeout_hours*3600-(time.monotonic()-start)
            if deadline<=0:raise TimeoutError('Series deadline')
            fresh=planning(args.n,args.groups,args.backend,resources,args.reserve_gib,threads)
            if fresh['initial_admission']!='READY_FOR_GUARDED_GROUPS':raise MemoryError('Fresh group preflight failed')
            groupdir.mkdir(exist_ok=True)
            worker_config=dict(config,lower_sizes=group,output=str(groupdir),timeout_seconds=deadline,
                cache_dir=str(Path(os.environ.get('MEANDER_CACHE_DIR',HERE/'.numba_cache')).resolve()))
            worker_config.pop('groups')
            save(groupdir/'CONFIG.json',worker_config)
            save(output/'STATE.json',dict(status='RUNNING',n=args.n,group=i,group_count=len(report['groups']),lower_sizes=group,completed_groups=i-1))
            print('Group %d/%d — lower sizes %s'%(i,len(report['groups']),group),flush=True)
            with(groupdir/'stdout.log').open('w',encoding='utf-8')as stdout:
                process=subprocess.Popen([sys.executable,str(HERE/'group_worker.py'),str(groupdir/'CONFIG.json')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,text=True,encoding='utf-8',errors='replace',bufsize=1)
                def tee(pipe):
                    for line in pipe:
                        stdout.write(line);stdout.flush();print(line.rstrip(),flush=True)
                    pipe.close()
                reader=threading.Thread(target=tee,args=(process.stdout,),daemon=True);reader.start()
                try:code=process.wait(timeout=deadline)
                except subprocess.TimeoutExpired:
                    process.terminate();process.wait();raise TimeoutError('Worker exceeded series deadline')
                finally:reader.join(timeout=5)
            process=None
            if code!=0:
                raise RuntimeError('Group %d stopped (exit %d). Details: %s. Completed groups are retained.'%(i,code,groupdir/'stdout.log'))
            r=json.loads(result_path.read_text());state=json.loads((groupdir/'STATE.json').read_text())
            if state.get('status')!='PASS'or state.get('result_sha256')!=sha(result_path):raise ArithmeticError('Worker receipt mismatch')
            records.append(r)
            save(output/'COMPLETED_GROUPS.json',dict(n=args.n,source_digest=report['source_digest'],groups=[dict(lower_sizes=v['lower_sizes'],contribution=v['Mn'],weighted_lower_coverage=v['weighted_lower_coverage'])for v in records]))
        value,covered=aggregate(args.n,records,report['groups'],report['source_digest'])
        known=json.loads((HERE/'KNOWN_RESULTS.json').read_text())
        if str(args.n)in known and str(value)!=known[str(args.n)]:raise ArithmeticError('Known-result mismatch')
        result=dict(status='PASS',n=args.n,Mn=str(value),weighted_lower_coverage=str(covered),source_digest=report['source_digest'],
            groups=report['groups'],backend=args.backend,threads=threads,
            numerical_seconds=sum(r['seconds']for r in records),worker_measured_wall_seconds=sum(r['measured_wall_seconds']for r in records),
            all_worker_launcher_seconds=sum(r['launcher_total_seconds']for r in records),controller_session_seconds=time.monotonic()-start,
            process_RSS_sampled_peak_bytes=max(r['process_RSS_sampled_peak_bytes']for r in records),
            process_RSS_note='Largest clean worker sampled peak; controller is a separate small process included in cgroup/host guards.',
            peak_coefficient_arena_bytes=max(r['peak_coefficient_arena_bytes']for r in records),
            peak_logical_bytes=max(r['peak_logical_bytes']for r in records),
            total_stages=sum(len(r['steps'])for r in records),
            groups_completed=len(records),matrix_read_bytes=0,matrix_write_bytes=0,independent_CRT=False,
            known_reference_checked=str(args.n)in known,
            group_results=[dict(group=i+1,path=str((output/('group_%02d'%(i+1))/'RESULT.json').relative_to(output)),sha256=sha(output/('group_%02d'%(i+1))/'RESULT.json'),lower_sizes=r['lower_sizes'],contribution=r['Mn'],seconds=r['seconds'],RSS_peak_bytes=r['process_RSS_sampled_peak_bytes'])for i,r in enumerate(records)])
        save(output/'RESULT.json',result);save(output/'STATE.json',dict(status='COMPLETE',n=args.n,Mn=str(value)))
        (output/'Mn.txt').write_text('M%d = %d\n'%(args.n,value),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True);return 0
    except BaseException as exc:
        if process is not None and process.poll()is None:
            process.terminate();process.wait()
        if output.exists():
            save(output/'ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            save(output/'STATE.json',dict(status='STOPPED',n=args.n,error=repr(exc),completed_groups=len(records)))
        raise
    finally:lock.close()

if __name__=='__main__':sys.exit(main())
