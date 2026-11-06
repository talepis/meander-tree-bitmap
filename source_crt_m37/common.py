"""Read-only resource planning and exact disjoint family metadata."""
import hashlib,json,math,os,sys,uuid
from pathlib import Path
from server_resources import Resources,GiB
HERE=Path(__file__).resolve().parent

def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex[:8]+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_digest():
    digest=hashlib.sha256()
    paths=[*HERE.glob('*.py'),*HERE.glob('*.sh'),HERE/'requirements.txt',HERE/'KNOWN_RESULTS.json',* (HERE/'tests').glob('*.py'),* (HERE/'engine64').glob('*.py')]
    for p in sorted(paths,key=lambda x:x.relative_to(HERE).as_posix()):
        digest.update(p.relative_to(HERE).as_posix().encode());digest.update(bytes.fromhex(sha(p)))
    return digest.hexdigest()

def ballot(m,h):
    if h<0 or h>m or (m-h)&1:return 0
    k=(m-h)//2
    return math.comb(m,k)-(math.comb(m,k-1)if k else 0)

def dimensions(n,backend='64'):
    if type(n)!=int or not 1<=n<=38:raise ValueError('Order must be 1..38')
    if backend not in ('32','64'):raise ValueError('Unknown rank backend')
    C=math.comb(2*n,n)//(n+1);maxB=max(ballot(m,h)for m in range(n+1)for h in range(m+1))
    maps=maxB+sum(ballot(m,h)for m in range(1,n)for h in range(2,m+1))
    width=4 if backend=='32' else 8
    if backend=='32'and(n>37 or maxB>=1<<32):raise OverflowError('32-bit rank domain exceeded')
    if maxB>=1<<60 or maps>=(1<<63)//width:raise OverflowError('Rank arena byte-offset domain')
    centroid_width=2 if n==1 else 2*(2*n//3)
    joint=max(ballot(a,h)*ballot(b,h)for a in range(n+1)for b in range(n+1)if a+b<=centroid_width for h in range(min(a,b)+1))
    if joint>=1<<63:raise OverflowError('Joint shape product domain')
    return dict(n=n,catalan=str(C),global_bound_decimal=str(C*C),global_bound_bits=(C*C).bit_length(),
        rank_dtype='uint32'if backend=='32'else'int64',rank_bytes=width,max_ballot_rank_count=maxB,
        closemap_elements=maps,closemap_bytes=width*maps,pooled_scratch_bytes=6*width*maxB,
        fixed_support_bytes=6*width*maxB+width*maps,centroid_width=centroid_width,max_joint_cell_domain=joint,
        stages_per_group=2*n-min(8,2*n),family_count=(n+1)//2,total_RSS_prediction_bytes=None,
        arithmetic='Certified per-block <=128 bits; checked rejection before target writes if cap fails; Python-integer final sum',
        matrix_files_allowed=False)

def groups_for(n,count):
    if type(count)!=int or not 1<=count<=4:raise ValueError('Use 1..4 groups for tests, 3/4 for M38')
    if n==38 and count not in (3,4):raise ValueError('M38 requires 3 or 4 groups')
    sizes=list(range(1,(n+1)//2+1));count=min(count,len(sizes))
    groups=[sizes[j::count]for j in range(count)]
    flat=sum(groups,[])
    if sorted(flat)!=sizes or len(flat)!=len(set(flat)):raise ArithmeticError('Invalid disjoint family partition')
    return groups

def configure(backend,threads,cache=None):
    os.environ.update(NUMBA_NUM_THREADS=str(threads),NUMBA_THREADING_LAYER='omp',OMP_NUM_THREADS=str(threads),
        OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        NUMBA_CACHE_DIR=str(Path(cache or os.environ.get('MEANDER_CACHE_DIR',HERE/'.numba_cache')).resolve()),MEANDER_GEOMETRY_ENGINE='bitmap')
    sys.path.insert(0,str(HERE/('engine'+backend)))

def choose_threads(value,resources):
    result=min(16,resources.default_threads)if value=='auto'else int(value)
    if not 1<=result<=resources.cpu_limit:raise ValueError('Threads exceed actual affinity/quota allocation')
    return result

def planning(n,count,backend,resources,reserve_gib,threads):
    dims=dimensions(n,backend);groups=groups_for(n,count);snap=resources.snapshot()
    reserve=int(reserve_gib*GiB);runtime=GiB
    workspace=snap['effective_available_bytes']-reserve-runtime-dims['fixed_support_bytes']
    return dict(n=n,groups=groups,backend=backend,dimensions=dims,resources=snap,threads=threads,
        reserve_bytes=reserve,runtime_allowance_bytes=runtime,workspace_budget_bytes=max(0,workspace),
        initial_admission='READY_FOR_GUARDED_GROUPS'if workspace>0 else'FIXED_SUPPORT_DOES_NOT_FIT',
        total_RSS_prediction_bytes=None,total_RSS_prediction_note='Full M38 group peaks are unmeasured. Admission is not a peak forecast; fresh per-allocation and RSS guards remain active.',
        source_digest=source_digest(),matrix_read_bytes=0,matrix_write_bytes=0,independent_CRT=False)

class JobLock:
    def __init__(self,path):
        self.f=open(path,'a+b')
        if self.f.tell()==0:self.f.write(b'0');self.f.flush()
        self.f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:self.f.close();raise
    def close(self):self.f.close()

def active_numeric_jobs():
    import psutil
    own=psutil.Process();ignore={own.pid,*[p.pid for p in own.parents()]};found=[]
    names={'server_meander.py','group_worker.py','native_worker.py','numeric_worker.py','worker.py'}
    for p in psutil.process_iter(['pid','name','cmdline','status']):
        try:
            if p.pid in ignore or p.info['status']==psutil.STATUS_ZOMBIE:continue
            args=p.info['cmdline']or[]
            if any(Path(a).name in names for a in args)and any('meander'in a.lower()for a in args):
                found.append(dict(pid=p.pid,name=p.info['name']))
        except(psutil.AccessDenied,psutil.NoSuchProcess):continue
    return found

def aggregate(n,records,groups,digest):
    if len(records)!=len(groups):raise ArithmeticError('Incomplete group set')
    total=coverage=0
    for r,g in zip(records,groups):
        if r.get('status')!='PASS'or r.get('n')!=n or r.get('lower_sizes')!=g or r.get('source_digest')!=digest:
            raise ArithmeticError('Group provenance or partition mismatch')
        if r.get('matrix_read_bytes')!=0 or r.get('matrix_write_bytes')!=0:raise ArithmeticError('Matrix disk I/O is forbidden')
        value=int(r['Mn']);covered=int(r['weighted_lower_coverage'])
        C=math.comb(2*n,n)//(n+1)
        if value<0 or covered<=0 or value>covered*C:raise ArithmeticError('Invalid group integer or partial Catalan-pair bound')
        total+=value;coverage+=covered
    C=math.comb(2*n,n)//(n+1)
    if coverage!=C or not 0<=total<=C*C:raise ArithmeticError('Catalan coverage or final bound failed')
    return total,coverage
