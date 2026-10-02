import unittest
import torch
from infer import perturb
from analyze import stratum_frame
class Tests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(1)
  self.x=torch.randn(3,9,776)
  self.data={'normalization':{'base':{'mean':torch.zeros(772),'std':torch.ones(772)},'force_delta':{'mean':torch.zeros(3),'std':torch.ones(3)}}}
 def test_bias(self):
  a=perturb(self.x,self.data,'force_bias_plus_0p1_fit_std');b=perturb(self.x,self.data,'force_bias_minus_0p1_fit_std')
  self.assertTrue(torch.allclose(a[:,:,769:772]-self.x[:,:,769:772],torch.full((3,9,3),.1),atol=1e-6))
  self.assertTrue(torch.equal(a[:,:,772:],self.x[:,:,772:]));self.assertTrue(torch.allclose((a+b)/2,self.x,atol=1e-6))
 def test_missing(self):
  a=perturb(self.x,self.data,'missing_base_penultimate_hold')
  self.assertTrue(torch.equal(a[:,7,:772],self.x[:,6,:772]));self.assertTrue(torch.equal(a[:,8,:772],self.x[:,8,:772]))
  self.assertTrue(torch.equal(a[:,5:,772:775],a[:,5:,769:772]-a[:,:4,769:772]))
  self.assertTrue(torch.equal(a[:,:,775],self.x[:,:,775]));self.assertTrue(torch.equal(a[:,:5,772:775],self.x[:,:5,772:775]))
 def test_singleclass(self):
  r=stratum_frame(None,[dict(metric_mask=True,target=1,p=.8)],'p',.5)
  self.assertEqual(r['frame_recall'],1);self.assertIsNone(r['frame_fpr']);self.assertIsNone(r['average_precision'])
if __name__=='__main__':unittest.main()
