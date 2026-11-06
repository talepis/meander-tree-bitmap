"""Portable cgroup fixture checks; these are not a real Linux execution."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import server_resources as R

class ResourceTests(unittest.TestCase):
    def test_ancestor_pressure_and_explicit_cap(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as td:
            assert Path(__file__).resolve().parent in Path(td).resolve().parents
            parent=Path(td); child=parent/'job'; child.mkdir()
            for path,limit,used in [(parent,90,81),(child,70,20)]:
                (path/'memory.max').write_text(str(limit*R.GiB))
                (path/'memory.current').write_text(str(used*R.GiB))
                (path/'cpu.max').write_text('250000 100000')
            paths=[('cgroup2',str(child),''),('cgroup2',str(parent),'')]
            vm=R.REAL_VM()._replace(total=200*R.GiB,available=150*R.GiB)
            with patch.object(R,'cgroup_paths',return_value=paths), patch.object(R,'REAL_VM',return_value=vm):
                resources=R.Resources()
                snap=resources.snapshot()
                self.assertEqual(snap['effective_total_bytes'],70*R.GiB)
                self.assertEqual(snap['effective_available_bytes'],9*R.GiB)
                self.assertLessEqual(resources.cpu_limit,2)
                (parent/'memory.current').write_text(str(88*R.GiB))
                self.assertEqual(resources.snapshot()['effective_available_bytes'],2*R.GiB)
                (parent/'memory.current').write_text(str(81*R.GiB))
                resources.memory_bytes=resources.snapshot()['process_RSS_bytes']+R.GiB
                self.assertLessEqual(resources.snapshot()['effective_available_bytes'],R.GiB)

    def test_namespace_root_and_mount_subtree(self):
        mount='1 0 0:1 /kubepods/job /sys/fs/cgroup rw - cgroup2 cgroup rw\n'
        self.assertEqual(R.cgroup_paths('0::/\n',mount),[('cgroup2',str(Path('/sys/fs/cgroup')),'')])
        self.assertEqual(R.cgroup_paths('0::/kubepods/job/sub\n',mount),[
            ('cgroup2',str(Path('/sys/fs/cgroup/sub')),''),('cgroup2',str(Path('/sys/fs/cgroup')),'')])

    def test_unknown_usage_rejects_finite_limit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as td:
            assert Path(__file__).resolve().parent in Path(td).resolve().parents
            p=Path(td); (p/'memory.max').write_text(str(10*R.GiB))
            with patch.object(R,'cgroup_paths',return_value=[('cgroup2',td,'')]):
                with self.assertRaises(RuntimeError):
                    R.Resources().snapshot()

if __name__=='__main__':
    unittest.main()
