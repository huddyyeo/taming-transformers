import torch
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore



class FIDMetric(FrechetInceptionDistance):
    def update(self, pred, gt):
        # convert to 255 uint8 as required by the module (as normalising has a bug)
        pred = (pred*255).to(torch.uint8).view(-1,*pred.shape[-3:])
        gt = (gt * 255).to(torch.uint8).view(-1,*gt.shape[-3:])
        super().update(pred,real=False)
        super().update(gt, real=True )

class InceptionMetric(InceptionScore):
    def update(self, pred, gt):
        del gt
        pred = (pred * 255).to(torch.uint8).view(-1, *pred.shape[-3:])
        super().update(pred)

class CodebookUsageMetric(torchmetrics.Metric):
    def __init__(self, codebook_size):
        super().__init__()
        self.codebook_size = codebook_size
        self.add_state('tokens', default=torch.zeros((self.codebook_size,)), dist_reduce_fx='sum')

    def update(self, tokens):
        new_tokens = torch.zeros_like(self.tokens)
        new_tokens[torch.unique(tokens)] = 1
        self.tokens += new_tokens

    def compute(self):
        n_unique_tokens = (self.tokens > 0).sum()
        return n_unique_tokens / self.codebook_size
