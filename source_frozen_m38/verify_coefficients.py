"""Compare 32/64-bit rank backends via RAM pipes. No array files are written."""
import argparse,json,multiprocessing as mp,sys,time,traceback
from pathlib import Path
from common import HERE,configure,save
CHUNK=4<<20

def child(connection,backend,n,threads):
    try:
        configure(backend,threads)
        import numpy as np,memory_release,target_geometry
        memory_release.configure('full');target_geometry.MODE='bitmap'
        from linear_support import calculate
        def layer(stage,data,state):
            arrays={'coefficients':data,'offsets':state['byte_offsets'],'widths':state['widths'],
                'shape':state['dshape'],'meta':state['meta'],'indices':state['indices'].astype(np.int64)}
            for axis,key in (('row','row_ids'),('column','column_ids')):
                for i,a in enumerate(state[key]):arrays[axis+'_%d'%i]=a.astype(np.int64)
            header=[(key,a.dtype.str,a.shape,a.nbytes)for key,a in arrays.items()]
            connection.send(('stage',stage,header))
            for a in arrays.values():
                view=memoryview(a).cast('B')
                for at in range(0,a.nbytes,CHUNK):connection.send_bytes(view[at:at+CHUNK])
        x=calculate(n,threads,tail=8,memory_mib=18000,timeout=1800,dense_cut=-125,minimize_grammar=True,
            split_maps=True,lifetime=True,geometry_mode='target',parallel_geometry=threads,numeric_mode='parallel',layer_check=layer)
        connection.send(('done',x['Mn']))
    except BaseException:connection.send(('error',traceback.format_exc()))
    finally:connection.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--orders',default='1,2,3,4,6,8,10,12,20,24,26');parser.add_argument('--threads',type=int,default=2);parser.add_argument('--output',type=Path,default=HERE/'COEFFICIENT_QA.json');args=parser.parse_args()
    context=mp.get_context('spawn');rows=[];counts=dict(coefficients=0,index_values=0,support_values=0,stages=0);started=time.perf_counter()
    for n in map(int,args.orders.split(',')):
        baseline={};answers=[]
        for backend in ('32','64'):
            receiver,sender=context.Pipe(duplex=False);process=context.Process(target=child,args=(sender,backend,n,args.threads));process.start();sender.close()
            try:
                while True:
                    message=receiver.recv()
                    if message[0]=='error':raise RuntimeError(message[1])
                    if message[0]=='done':answers.append(message[1]);break
                    _,stage,header=message
                    if backend=='32':baseline[stage]=dict(header=header,arrays={})
                    else:assert header==baseline[stage]['header'],(n,stage,'array layout')
                    for key,dtype,shape,nbytes in header:
                        if backend=='32':buf=bytearray(nbytes)
                        else:buf=baseline[stage]['arrays'][key]
                        for at in range(0,nbytes,CHUNK):
                            chunk=receiver.recv_bytes();assert len(chunk)==min(CHUNK,nbytes-at)
                            if backend=='32':buf[at:at+len(chunk)]=chunk
                            else:assert memoryview(buf)[at:at+len(chunk)]==chunk,(n,stage,key,at)
                        if backend=='32':baseline[stage]['arrays'][key]=buf
                        elif key=='indices':counts['index_values']+=nbytes//8
                        elif key.startswith(('row_','column_')):counts['support_values']+=nbytes//8
                    if backend=='64':
                        import numpy as np
                        shape=np.frombuffer(baseline[stage]['arrays']['shape'],dtype=np.int64).reshape(-1,2)
                        counts['coefficients']+=sum(int(a)*int(b)for a,b in shape);counts['stages']+=1
                        del baseline[stage]
            finally:
                receiver.close();process.join(timeout=5)
                if process.is_alive():process.terminate();process.join()
            assert process.exitcode==0
        assert answers[0]==answers[1]
        rows.append(dict(n=n,Mn=answers[0],status='EXACT_ARRAY_EQUALITY_PASS'));print(json.dumps(rows[-1]),flush=True)
    record=dict(status='PASS',seconds=time.perf_counter()-started,counts=counts,cases=rows,
        comparison='Direct byte equality for all coefficient bytes, offsets, widths, shapes, metadata, and normalized signed64 index/support values.',
        transport='RAM-only chunked pipes; reference snapshots retained in parent RAM; one numerical child at a time.',matrix_files_written=False)
    save(args.output,record)

if __name__=='__main__':main()
