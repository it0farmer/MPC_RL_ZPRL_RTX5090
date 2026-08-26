import unittest, torch
from mpcrl.lewm import SIGReg
class T(unittest.TestCase):
    def test_finite_and_grad(self):
        torch.manual_seed(0); z=torch.randn(64,8,requires_grad=True); loss=SIGReg(knots=9,num_proj=32)(z); self.assertTrue(torch.isfinite(loss)); loss.backward(); self.assertIsNotNone(z.grad)
if __name__=='__main__': unittest.main()
