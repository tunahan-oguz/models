import torch
import torch.nn as nn
import time

class InitialLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, 2, padding=3)
        self.mp = nn.MaxPool2d((2, 2), stride=2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv(x))
        return self.mp(x)

class SecondLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(64, 192, 3, padding=1)
        self.mp = nn.MaxPool2d((2, 2), stride=2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv(x))
        return self.mp(x)

class ThirdLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(192, 128, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.mp = nn.MaxPool2d((2, 2), stride=2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.conv4(x)
        return self.mp(x)

class FourthLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            *([nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU()] * 4)
        )
        self.conv2 = nn.Conv2d(512, 512, 1)
        self.conv3 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mp = nn.MaxPool2d((2, 2), stride=2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return self.mp(x)

class FifthLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            *([nn.Conv2d(1024, 512, 1), 
               nn.ReLU(), 
               nn.Conv2d(512, 1024, 3, padding=1)] * 2)
        )
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv3 = nn.Conv2d(1024, 1024, 3, stride=2, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.act(self.conv3(x))




class SixthLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        return self.act(self.conv2(x))

class YOLO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = InitialLayer()
        self.layer2 = SecondLayer()
        self.layer3 = ThirdLayer()
        self.layer4 = FourthLayer()
        self.layer5 = FifthLayer()
        self.layer6 = SixthLayer()
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 7 * 7 * 30)
        )
        self.act = nn.ReLU()


    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = self.act(self.layer4(x))
        x = self.act(self.layer5(x))
        x = self.act(self.layer6(x))
        x = x.flatten(1)
        x = self.fc(x)
        return x.reshape((-1, 7, 7, 30))


yolo = YOLO()
yolo.cuda()
s = time.time()
for i in range(60):
    x = torch.randn(8, 3, 448, 448).cuda()
    yolo(x)
e = time.time()
print(e - s)
