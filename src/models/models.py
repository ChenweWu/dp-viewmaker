import timm
import torch
import torchvision
import torch.nn as nn


class Resnet200D(nn.Module):

    def __init__(self, model_name = 'resnet200d', number_of_classes = 2):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.model = timm.create_model(model_name,pretrained=True)
        self.in_dims = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc = nn.Linear(self.in_dims, self.number_of_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.model.global_pool = nn.Identity()


    def forward(self,x):
        in_x = self.model(x)
        pooling_x = self.pool(in_x).view(x.shape[0],-1)
        output = self.fc(pooling_x)
        return output


if __name__ == "__main__":
    model_test = Resnet200D(model_name='resnet200d',number_of_classes=2)
    print(model_test)
