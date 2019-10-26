
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 100)
    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128*26*26, 2048)
        self.fc2 = nn.Linear(2048, 100)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128*26*26)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VGGLikeModel(nn.Module):
    def __init__(self):
        super(VGGLikeModel, self).__init__()
        self.conv11 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv21 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 100)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)


    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.max_pool2d(x, (2, 2))
        x = self.dropout1(x)

        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv22(x))
        x = F.max_pool2d(x, (2, 2))
        x = self.dropout1(x)

        x = x.view(-1, 64 * 8 * 8)

        x = F.relu(self.fc1(x))
        # x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class WideModel(nn.Module):
    def __init__(self):
        super(WideModel, self).__init__()
                                                                            # input ->  3, 32, 32
        self.fx_conv = nn.Conv2d(3, 64, 1)                                  #  3, 32, 32 -> 64, 32, 32

                                                                            # wide blocks
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, padding_mode='same')   # 64, 32, 32 -> 64, 16, 16
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2, padding_mode='same')   # 64, 32, 32 -> 64, 16, 16
        self.conv3 = nn.Conv2d(64, 64, 7, padding=3, padding_mode='same')   # 64, 32, 32 -> 64, 16, 16
                                                                            # concat 3 x (64, 16, 16) -> 64*3, 16, 16
        self.comb_conv1 = nn.Conv2d(64*3, 128, 1)                           # 64*3, 16, 16 -> 128, 16, 16
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='same') # 128, 16, 16 -> 128,  8,  8
        self.conv5 = nn.Conv2d(128, 128, 5, padding=2, padding_mode='same') # 128, 16, 16 -> 128,  8,  8
        self.conv6 = nn.Conv2d(128, 128, 7, padding=3, padding_mode='same') # 128, 16, 16 -> 128,  8,  8
                                                                            # concat 3 x (128,  8,  8) -> 128*3,  8,  8
        self.comb_conv2 = nn.Conv2d(128*3, 256, 1)                          # 128*3,  8,  8 -> 256,  8,  8

        self.pool = nn.MaxPool2d((2, 2))                                    # to each conv
        self.drop = nn.Dropout2d(0.25)                                      # to each conv

                                                                            # prediction layers
        self.ave_pool = nn.AdaptiveAvgPool2d((1, 1))                        # 256,  8,  8 -> 256
        self.fc1 = nn.Linear(256, 512)                                      # 256 -> 512
        self.fc2 = nn.Linear(512, 100)                                      # 512 -> 100
        
        self.drop2 = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.fx_conv(x))

        wide1 = self.pool(F.relu(self.conv1(x)))
        wide1 = self.drop(wide1)
        wide2 = self.pool(F.relu(self.conv2(x)))
        wide2 = self.drop(wide2)
        wide3 = self.pool(F.relu(self.conv3(x)))
        wide3 = self.drop(wide3)

        x = torch.cat([wide1, wide2, wide3], dim=1)
        x = F.relu(self.comb_conv1(x))

        wide4 = self.pool(F.relu(self.conv4(x)))
        wide4 = self.drop(wide4)
        wide5 = self.pool(F.relu(self.conv5(x)))
        wide5 = self.drop(wide5)
        wide6 = self.pool(F.relu(self.conv6(x)))
        wide6 = self.drop(wide6)

        x = torch.cat([wide4, wide5, wide6], dim=1)
        x = F.relu(self.comb_conv2(x))

        x = self.ave_pool(x)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

if __name__=='__main__':
    model = WideModel()
    print(model)