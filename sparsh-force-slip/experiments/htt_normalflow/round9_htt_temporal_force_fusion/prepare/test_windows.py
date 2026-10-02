import unittest
import numpy as np
from export import windows
class TestWindows(unittest.TestCase):
 def test_exact_history_and_difference(self):
  v=np.tile(np.arange(30)[:,None],(1,192)).astype(np.float32);f=np.tile(np.arange(30)[:,None],(1,3)).astype(np.float32)
  t,x=windows(v,f);self.assertEqual(t[0],13);self.assertEqual(x.shape,(17,9,199));np.testing.assert_array_equal(x[0,:,0],np.arange(5,14));np.testing.assert_array_equal(x[0,:,192],np.arange(5,14));self.assertTrue((x[:,:5,195:]==0).all());self.assertTrue((x[:,5:,195:198]==5).all());self.assertTrue((x[:,5:,198]==1).all())
 def test_future_invariance(self):
  rng=np.random.default_rng(12);v=rng.normal(size=(30,192)).astype(np.float32);f=rng.normal(size=(30,3)).astype(np.float32);_,x=windows(v,f);v[14:]=999;f[14:]=-999;_,y=windows(v,f);np.testing.assert_array_equal(x[0],y[0])
 def test_visual_force_separation(self):
  v=np.ones((20,192),np.float32);f=np.ones((20,3),np.float32);_,x=windows(v,f);_,y=windows(v,f*98);np.testing.assert_array_equal(x[:,:,:192],y[:,:,:192])
 def test_shape_rejected(self):
  with self.assertRaises(ValueError):windows(np.zeros((20,193)),np.zeros((20,3)))
if __name__=='__main__':unittest.main()
