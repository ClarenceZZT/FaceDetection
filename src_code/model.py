import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
import torch.nn.utils.prune as prune

class FaceKeypointResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad, pruning_amount = 0):
        super(FaceKeypointResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
        self.newFC = nn.Linear(2048, 136)
        if pruning_amount != 0:
            self.prune_network(pruning_amount)    
        
    def prune_network(self, pruning_amount):
        # Iterate over all modules and prune convolutional layers
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, 'weight', amount=pruning_amount)
                
    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        res = self.newFC(x)
        return res