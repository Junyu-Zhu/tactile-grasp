import unittest,numpy as np
from diagnostics import correctness,residual_stats,sequence_bin
class Tests(unittest.TestCase):
 def test_correction_and_harm(self):
  r=correctness([0,0,2,2,1],[.8,.2,.2,.8,.8],[.2,.8,.8,.2,.2],.5)
  np.testing.assert_equal(r['corrected'],[1,0,1,0,0]);np.testing.assert_equal(r['harmed'],[0,1,0,1,0])
 def test_never_alarm(self):
  r=correctness([0,2],[1,1],[1,1],np.nextafter(1.,np.inf));self.assertFalse(r['corrected'].any());self.assertFalse(r['harmed'].any())
 def test_saturation(self):
  r=residual_stats([-2,-1.9,0,.1,2],2);self.assertEqual(r['saturated'],3);self.assertEqual(r['negative'],2)
 def test_bad_bound(self):
  with self.assertRaises(ValueError):residual_stats([2.1],2)
 def test_bins(self):self.assertEqual([sequence_bin(t,101) for t in (0,25,50,75,100)],[0,1,2,3,3])
if __name__=='__main__':unittest.main()
