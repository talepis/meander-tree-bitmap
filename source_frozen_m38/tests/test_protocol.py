import copy,math,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import GiB,JobLock,aggregate,dimensions,groups_for,planning

class ProtocolTests(unittest.TestCase):
    def test_m38_exact_rank_and_bound_domain(self):
        d=dimensions(38)
        self.assertEqual(d['max_ballot_rank_count'],6768687870)
        self.assertEqual(d['rank_dtype'],'int64')
        self.assertEqual(d['global_bound_bits'],135)
        self.assertEqual(d['stages_per_group'],68)
        self.assertEqual(d['fixed_support_bytes'],8*(6*d['max_ballot_rank_count']+d['closemap_elements']))
        self.assertLess(d['max_joint_cell_domain'],1<<63)
    def test_unsafe_old_domain_rejected(self):
        with self.assertRaises(OverflowError):dimensions(38,'32')
        with self.assertRaises(ValueError):dimensions(39)
    def test_partition_all_small_and_target_orders(self):
        for n in [*range(1,14),28,31,37,38]:
            for k in (3,4):
                groups=groups_for(n,k);flat=sum(groups,[])
                self.assertEqual(sorted(flat),list(range(1,(n+1)//2+1)))
                self.assertEqual(len(flat),len(set(flat)))
    def test_m38_fewer_groups_rejected(self):
        for k in (1,2):
            with self.assertRaises(ValueError):groups_for(38,k)
    def test_real_available_not_nominal_host_total(self):
        class Fake:
            def snapshot(self):return dict(effective_total_bytes=1900*GiB,effective_available_bytes=100*GiB)
        p=planning(38,4,'64',Fake(),16,16)
        self.assertEqual(p['initial_admission'],'FIXED_SUPPORT_DOES_NOT_FIT')
        self.assertIsNone(p['total_RSS_prediction_bytes'])
    def records(self):
        n=38;groups=groups_for(n,3);C=math.comb(76,38)//39
        cover=[C//3,C//3,C-2*(C//3)]
        records=[dict(status='PASS',n=38,lower_sizes=g,source_digest='test',Mn=str(1<<128),weighted_lower_coverage=str(c),matrix_read_bytes=0,matrix_write_bytes=0)for g,c in zip(groups,cover)]
        return records,groups
    def test_sum_above_128_bits_is_exact_python_integer(self):
        records,groups=self.records();value,coverage=aggregate(38,records,groups,'test')
        self.assertEqual(value,3*(1<<128));self.assertEqual(coverage,math.comb(76,38)//39)
    def test_incomplete_wrong_source_or_overlap_rejected(self):
        records,groups=self.records()
        with self.assertRaises(ArithmeticError):aggregate(38,records[:-1],groups,'test')
        for field,value in [('source_digest','changed'),('lower_sizes',groups[1]),('matrix_write_bytes',1),('weighted_lower_coverage','1'),('Mn',str(1<<160))]:
            bad=copy.deepcopy(records);bad[0][field]=value
            with self.assertRaises(ArithmeticError):aggregate(38,bad,groups,'test')
    def test_one_parent_lock(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)as td:
            lock=JobLock(Path(td)/'job.lock')
            try:
                with self.assertRaises(OSError):JobLock(Path(td)/'job.lock')
            finally:lock.close()

if __name__=='__main__':unittest.main()
