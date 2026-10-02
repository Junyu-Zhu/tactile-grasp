import unittest,ast
from pathlib import Path
# Load pure helper only: local test does not import server plotting dependencies.
s=ast.parse(Path(__file__).with_name('case_analysis.py').read_text());f=next(x for x in s.body if isinstance(x,ast.FunctionDef) and x.name=='class_rates');ns={};exec(compile(ast.Module(body=[f],type_ignores=[]),'helper','exec'),ns);rates=ns['class_rates']
class Tests(unittest.TestCase):
 def test_no_gross(self):
  r=rates(dict(tn=2,fp=3,tp=0,fn=0));self.assertIsNone(r['gross_miss']);self.assertFalse(r['gross_applicable']);self.assertEqual(r['static_fpr'],.6)
 def test_no_static(self):
  r=rates(dict(tn=0,fp=0,tp=3,fn=1));self.assertIsNone(r['static_fpr']);self.assertEqual(r['gross_miss'],.25)
if __name__=='__main__':unittest.main()
