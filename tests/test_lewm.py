import unittest, torch
from mpcrl.lewm import LeWorldModel
class T(unittest.TestCase):
    def test_loss_backward(self):
        torch.manual_seed(0); model=LeWorldModel(action_dim=2,latent_dim=16,hidden=32,sigreg_projections=8,sigreg_knots=5); f=torch.randint(0,256,(8,3,32,32),dtype=torch.uint8); nf=torch.randint(0,256,(8,3,32,32),dtype=torch.uint8); a=torch.randn(8,2); loss,info=model.loss(f,a,nf); self.assertTrue(torch.isfinite(loss)); loss.backward(); self.assertIn('sigreg',info)
if __name__=='__main__': unittest.main()
