##########################################################################
# Copyright 2022 Jianping Cai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################
# Models for federated learning
##########################################################################
import torch
import torch.nn as nn

class IotCNN(nn.Module):
    def __init__(self, channels=[9,16,32], dim_hiddens=[3712,1024,512], num_classes=13, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hiddens[0], dim_hiddens[1]),
            nn.ReLU(),
            nn.Linear(dim_hiddens[1], dim_hiddens[2]),
            nn.ReLU(),
            nn.Linear(dim_hiddens[2], num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def CNN_PAMAP():
    params = {'channels': [27,32,32], 'dim_hiddens': [44*32,512,64], 'num_classes': 10}
    return IotCNN(**params)

def CNN_HAR():
    params = {'channels': [9,32,32], 'dim_hiddens': [26*32,512,64], 'num_classes': 6}
    return IotCNN(**params)