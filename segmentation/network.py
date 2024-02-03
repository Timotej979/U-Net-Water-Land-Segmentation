import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(UNet, self).__init__()
        
        # Encoder blocks
        self.block1 = nn.Sequential()
        self.block1.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1))
        self.block1.add_module('relu1', nn.ReLU())   
        self.block1.add_module('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
        self.block1.add_module('relu2', nn.ReLU())

        self.block2 = nn.Sequential()
        self.block2.add_module('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.block2.add_module('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
        self.block2.add_module('relu3', nn.ReLU())
        self.block2.add_module('conv4', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
        self.block2.add_module('relu4', nn.ReLU())

        self.block3 = nn.Sequential()
        self.block3.add_module('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.block3.add_module('conv5', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
        self.block3.add_module('relu5', nn.ReLU())
        self.block3.add_module('conv6', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.block3.add_module('relu6', nn.ReLU())

        self.block4 = nn.Sequential()
        self.block4.add_module('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))
        self.block4.add_module('conv7', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1))
        self.block4.add_module('relu7', nn.ReLU())
        self.block4.add_module('conv8', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.block4.add_module('relu8', nn.ReLU())
        
        self.block5 = nn.Sequential()
        self.block5.add_module('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))
        self.block5.add_module('conv9', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1))
        self.block5.add_module('relu9', nn.ReLU())
        self.block5.add_module('conv10', nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1))
        self.block5.add_module('relu10', nn.ReLU())
        self.block5.add_module('conv11', nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2))
        self.block5.add_module('relu10', nn.ReLU())

        # Decoder blocks
        self.block6 = nn.Sequential()
        self.block6.add_module('conv12', nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1))
        self.block6.add_module('relu12', nn.ReLU())
        self.block6.add_module('conv13', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        self.block6.add_module('relu13', nn.ReLU())
        self.block6.add_module('conv14', nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2))
        self.block6.add_module('relu14', nn.ReLU())

        self.block7 = nn.Sequential()
        self.block7.add_module('conv15', nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1))
        self.block7.add_module('relu15', nn.ReLU())
        self.block7.add_module('conv16', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.block7.add_module('relu16', nn.ReLU())
        self.block7.add_module('conv17', nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2))
        self.block7.add_module('relu17', nn.ReLU())

        self.block8 = nn.Sequential()
        self.block8.add_module('conv18', nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1))
        self.block8.add_module('relu18', nn.ReLU())
        self.block8.add_module('conv19', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
        self.block8.add_module('relu19', nn.ReLU())
        self.block8.add_module('conv20', nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2))
        self.block8.add_module('relu20', nn.ReLU())
  
        self.block9 = nn.Sequential()
        self.block9.add_module('conv21', nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1))
        self.block9.add_module('relu21', nn.ReLU())
        self.block9.add_module('conv22', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
        self.block9.add_module('relu22', nn.ReLU())
        self.block9.add_module('conv23', nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)) # Conv1-1
        

    def forward(self, x):
        # Encoder
        features1 = self.block1(x) 
        features2 = self.block2(features1)
        features3 = self.block3(features2)
        features4 = self.block4(features3)
        features5 = self.block5(features4)

        # Decoder
        concat5 = torch.cat([features4, features5], dim=1)
        features6 = self.block6(concat5)
        concat6 = torch.cat([features3, features6], dim=1)
        features7 = self.block7(concat6)
        concat7 = torch.cat([features2, features7], dim=1)
        features8 = self.block8(concat7)
        concat8 = torch.cat([features1, features8], dim=1)
        mask = self.block9(concat8)

        # Permutate mask from [bN,ch,w,h] to [bN,w,h,ch]
        mask = torch.permute(mask, (0, 2, 3, 1))

        return mask