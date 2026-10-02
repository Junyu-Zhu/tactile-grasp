import unittest,numpy as np
from compare_models import compare
class CompareTests(unittest.TestCase):
 def test_all_types(self):
  r=compare([0,0,2,2,0,2,1],[.9,.1,.1,.9,.9,.1,.9],[.1,.9,.9,.1,.9,.1,.1],.5,.5)
  for k in ('static_corrected','static_harmed','gross_corrected','gross_harmed','common_static_error','common_gross_error','incipient_excluded'):self.assertEqual(r[k],1)
 def test_different_thresholds(self):
  r=compare([0,2],[.6,.6],[.6,.6],.5,.7);self.assertEqual(r['static_corrected'],1);self.assertEqual(r['gross_harmed'],1)
 def test_never_alarm(self):
  r=compare([0,2],[1,1],[1,1],np.nextafter(1.,np.inf),np.nextafter(1.,np.inf));self.assertEqual(r['common_gross_error'],1);self.assertEqual(r['static_harmed'],0)
if __name__=='__main__':unittest.main()
