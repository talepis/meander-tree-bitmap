"""One independent family group; coefficients are anonymous RAM arrays only."""
import gc,json,os,platform,sys,threading,time,traceback
from pathlib import Path
from common import HERE,Resources,GiB,configure,save,sha,source_digest,dimensions

def main():
    config=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(config['output']).resolve()
    modulus=int(config['modulus_decimal']);ring=config['ring']
    if (ring,modulus) not in (('pow2',1<<64),('ones',(1<<64)-1),('seven',7)):
        raise ValueError('Unsupported CRT ring')
    if config['source_digest']!=source_digest():raise RuntimeError('Worker source digest mismatch')
    if (out/'RESULT.json').exists():raise FileExistsError('A group result is already present')
    out.mkdir(parents=True,exist_ok=True)
    resources=Resources(config.get('memory_gib'));reserve=config['reserve_bytes'];threads=config['threads']
    dims=dimensions(config['n'],config['backend']);initial=resources.snapshot()
    workspace=initial['effective_available_bytes']-reserve-GiB-dims['fixed_support_bytes']
    if workspace<=0:raise MemoryError('Fixed support and reserve do not fit')
    app_limit=initial['process_RSS_bytes']+max(0,initial['effective_available_bytes']-reserve)
    stop=threading.Event();peak=[initial['process_RSS_bytes']];ordinary_start=None
    save(out/'PREFLIGHT.json',dict(resources=initial,dimensions=dims,reserve_bytes=reserve,workspace_budget_bytes=workspace,process_RSS_limit_bytes=app_limit,source_digest=config['source_digest']))
    save(out/'STATE.json',dict(status='PREPARING',n=config['n'],lower_sizes=config['lower_sizes'],pid=os.getpid()))
    def watch():
        import psutil
        last=0
        try:
            while not stop.wait(.005):
                rss=psutil.Process().memory_info().rss;peak[0]=max(peak[0],rss)
                now=time.monotonic()
                if now-last<.025:continue
                last=now;snap=resources.snapshot()
                if rss>app_limit or snap['effective_available_bytes']<min(8*GiB,reserve//2):
                    error=dict(status='STOPPED_RESOURCE',process_RSS_bytes=rss,process_RSS_limit_bytes=app_limit,resources=snap)
                    save(out/'RESOURCE_STOP.json',error);save(out/'STATE.json',error);os._exit(75)
        except BaseException as e:
            save(out/'RESOURCE_STOP.json',dict(status='GUARD_ERROR',error=repr(e)));os._exit(76)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start();runstart=time.perf_counter()
    try:
        configure(config['backend'],threads,config.get('cache_dir'));resources.install_guard(reserve)
        import numba,numpy,psutil,memory_release,target_geometry
        target_geometry.MODE='bitmap';memory_release.configure('full')
        from linear_support import calculate
        kwargs=dict(workers=threads,tail=8,memory_mib=workspace/(1<<20),timeout=config['timeout_seconds'],dense_cut=-125,
            minimize_grammar=True,split_maps=True,lifetime=True,geometry_mode='target',parallel_geometry=threads,numeric_mode='modular',modulus=modulus,index_prealloc_bytes=0)
        warm_seconds=None
        if config.get('warmup_n'):
            t=time.perf_counter();warm=calculate(config['warmup_n'],**kwargs);warm_seconds=time.perf_counter()-t
            if int(warm['Mn'])!=int(json.loads((HERE/'KNOWN_RESULTS.json').read_text())[str(config['warmup_n'])])%modulus:raise ArithmeticError('Warm-up mismatch')
            del warm;gc.collect();peak[0]=psutil.Process().memory_info().rss
        measured_start=time.perf_counter();imports=measured_start-runstart;last_message=[0]
        try:ordinary_start=psutil.Process().io_counters()._asdict()
        except(AttributeError,psutil.Error):pass
        with (out/'STAGES.jsonl').open('w',encoding='utf-8')as log:
            def progress(row):
                snap=resources.snapshot()
                record=dict(row,process_RSS_bytes=snap['process_RSS_bytes'],process_RSS_sampled_peak_bytes=peak[0],effective_available_bytes=snap['effective_available_bytes'],matrix_read_bytes=0,matrix_write_bytes=0)
                log.write(json.dumps(record)+'\n');log.flush()
                save(out/'STATE.json',dict(status='RUNNING',n=config['n'],lower_sizes=config['lower_sizes'],stage=record))
                now=time.monotonic()
                if now-last_message[0]>=10:
                    print(json.dumps(dict(stage=row['stage'],elapsed=row['elapsed'],RSS_GiB=round(snap['process_RSS_bytes']/GiB,3))),flush=True);last_message[0]=now
            if config['backend']!='64':raise ValueError('CRT requires rank-64 family backend')
            result=calculate(config['n'],lower_sizes=config['lower_sizes'],progress=progress,**kwargs)
        if not 0<=int(result['Mn'])<modulus:raise ArithmeticError('Residue outside selected modulus')
        measured=time.perf_counter()-measured_start
        if len(result['steps'])!=dims['stages_per_group']:raise ArithmeticError('Incomplete stage history')
        result.update(source_digest=config['source_digest'],backend=config['backend'],rank_dtype=dims['rank_dtype'],ring=ring,
            process_RSS_sampled_peak_bytes=peak[0],RSS_sampling_period_seconds=.005,
            launcher_total_seconds=time.perf_counter()-runstart,measured_wall_seconds=measured,
            imports_and_optional_warmup_seconds=imports,warmup_seconds=warm_seconds,
            compilation_seconds=None,compilation_note='First JIT/cache load remains in recorded phases; no estimated subtraction.',
            python=sys.version,platform=platform.platform(),numpy=numpy.__version__,numba=numba.__version__,threading_backend=numba.threading_layer(),
            workspace_budget_bytes=workspace,process_RSS_limit_bytes=app_limit,reserve_bytes=reserve,initial_resources=initial,
            independent_CRT=False,matrix_read_bytes=0,matrix_write_bytes=0)
        try:result['ordinary_process_IO_end']=psutil.Process().io_counters()._asdict()
        except(AttributeError,psutil.Error):result['ordinary_process_IO_end']=None
        result['ordinary_process_IO_start']=ordinary_start
        try:
            import resource
            result['process_RSS_OS_high_water_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        except ImportError:result['process_RSS_OS_high_water_bytes']=None
        save(out/'RESULT.json',result)
        save(out/'STATE.json',dict(status='PASS',n=config['n'],lower_sizes=config['lower_sizes'],Mn=result['Mn'],result_sha256=sha(out/'RESULT.json')))
        print(json.dumps(dict(status='PASS',contribution=result['Mn'],seconds=measured,RSS_bytes=peak[0])),flush=True)
        return 0
    except BaseException as e:
        save(out/'ERROR.json',dict(status='ERROR',error=repr(e),traceback=traceback.format_exc()))
        save(out/'STATE.json',dict(status='ERROR',error=repr(e)));raise
    finally:stop.set();watcher.join(timeout=1)

if __name__=='__main__':sys.exit(main())
