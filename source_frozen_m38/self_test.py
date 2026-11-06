#!/usr/bin/env python3
import argparse,gc,json,math,platform,sys,time
from common import HERE,Resources,choose_threads,configure,groups_for,save,source_digest

def brute(n):
    pairs=[]
    def rec(i,stack,p):
        if i==2*n:pairs.append(tuple(p));return
        if len(stack)<2*n-i:rec(i+1,stack+[i],p)
        if stack:
            j=stack[-1];q=p.copy();q[i]=j;q[j]=i;rec(i+1,stack[:-1],q)
    rec(0,[],[-1]*(2*n));count=0
    for lower in pairs:
        for upper in pairs:
            v=k=0
            while True:
                v=upper[lower[v]];k+=1
                if not v:break
            count+=k==n
    return count,len(pairs)**2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--threads',default='auto');parser.add_argument('--output',default=str(HERE/'SELF_TEST_RESULT.json'));args=parser.parse_args()
    if not __debug__:raise RuntimeError('Do not run tests with Python -O')
    from verify_bundle import main as verify
    verify();resources=Resources();threads=choose_threads(args.threads,resources)
    configure('64',threads)
    import memory_release,target_geometry
    memory_release.configure('full');target_geometry.MODE='bitmap'
    from linear_support import calculate
    from deterministic import Compiler
    from weighted import Operators
    from rank_tests import run as rank_run
    started=time.perf_counter();boundary=rank_run()
    known=json.loads((HERE/'KNOWN_RESULTS.json').read_text());rows=[];pairchecks=0
    flags=dict(workers=threads,tail=8,memory_mib=2048,timeout=1800,dense_cut=-125,minimize_grammar=True,split_maps=True,lifetime=True,geometry_mode='target',parallel_geometry=threads,numeric_mode='parallel')
    for n in range(1,13):
        if n<=6:
            value,pairs=brute(n);assert str(value)==known[str(n)];pairchecks+=pairs
        for k in (3,4):
            parts=[calculate(n,lower_sizes=g,**flags)for g in groups_for(n,k)]
            answer=sum(int(p['Mn'])for p in parts);coverage=sum(int(p['weighted_lower_coverage'])for p in parts)
            assert str(answer)==known[str(n)]and coverage==math.comb(2*n,n)//(n+1),(n,k,answer)
            assert all(p['matrix_read_bytes']==p['matrix_write_bytes']==0 for p in parts)
            rows.append(dict(n=n,requested_groups=k,effective_groups=len(parts),Mn=str(answer),status='PASS'))
        print('SELF TEST M%d PASS'%n,flush=True);gc.collect()
    grammar=[]
    for k in (3,4):
        coverage=0;parts=[]
        for g in groups_for(38,k):
            compiler=Compiler(38,g,Operators(8));covered=compiler.lower_count((0,0,0,0,0,0));compiled=compiler.compile()
            coverage+=covered;parts.append(dict(lower_sizes=g,nodes=compiled['nodes'],weighted_coverage=str(covered)))
        assert coverage==math.comb(76,38)//39
        grammar.append(dict(groups=k,coverage=str(coverage),parts=parts))
    record=dict(status='PASS',source_digest=source_digest(),rank64_boundary_status=boundary['status'],boundary=boundary,
        whole_small_orders=rows,brute_pair_checks=pairchecks,M38_real_grammar_checks=grammar,
        threads=threads,platform=platform.platform(),python=sys.version,seconds=time.perf_counter()-started,
        full_M38_numerical_run=False,independent_CRT=False)
    save(args.output,record);print(json.dumps(dict(status='SELF_TEST_PASS',seconds=record['seconds'],output=args.output)),flush=True)

if __name__=='__main__':main()
