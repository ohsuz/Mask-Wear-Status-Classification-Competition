import torch.nn as nn

class FoldEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD, modelE):
        super(FoldEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE
        
    def forward(self, inputs):
        x1 = self.modelA(inputs)
        x2 = self.modelB(inputs)
        x3 = self.modelC(inputs)
        x4 = self.modelD(inputs)
        x5 = self.modelE(inputs)
        x = x1 + x2 + x3 + x4 + x5
        x = nn.Softmax(dim=1)(x)
        return x