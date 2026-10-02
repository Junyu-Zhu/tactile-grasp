import unittest
import torch
from benchmark import assemble
class Test(unittest.TestCase):
 def test_matches_raw_history(self):
  bases=[(torch.full((1,192),float(i)),torch.full((1,3),float(i*i))) for i in range(9)];norm={'mean':torch.zeros(198),'std':torch.ones(198)}
  raw,x=assemble(bases,norm,'F_delta');self.assertEqual(raw.shape,(1,9,199));self.assertTrue(torch.equal(raw,x));self.assertEqual(float(x[0,5,195]),25.);self.assertEqual(float(x[0,8,195]),55.);self.assertEqual(float(x[0,:5,195:].sum()),0.)
 def test_visual_requires_no_force(self):
  bases=[(torch.ones((1,192)),None) for _ in range(9)];norm={'mean':torch.zeros(198),'std':torch.ones(198)};raw,x=assemble(bases,norm,'V_temporal');self.assertEqual(float(raw[:,:,192:198].abs().sum()),0.);self.assertEqual(float(x[:,:,198].sum()),4.)
 def test_invalid_delta_stays_zero_after_normalizing(self):
  bases=[(torch.ones((1,192)),torch.ones((1,3))) for _ in range(9)];norm={'mean':torch.ones(198),'std':torch.ones(198)};raw,x=assemble(bases,norm,'F_history');self.assertEqual(float(x[:,:5,195:198].abs().sum()),0.)
if __name__=='__main__':unittest.main()
