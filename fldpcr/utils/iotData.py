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
# Data for HAR and PAMAP
##########################################################################
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import py7zr

class PamapDataset(Dataset):
    def __init__(self, root='data', train=True, transform=None):
        folder_path=root+'/PAMAP/'
        archive_path = root+'/PAMAP.7z'

        if not os.path.exists(folder_path):
            filenames = [archive_path+".{:03}".format(i+1) for i in range(4)]
            tmp_path=root+'/tmp.7z'
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"Extracting {archive_path} to {root}...")
            with open(tmp_path, 'ab') as outfile:  # append in binary mode
                for fname in filenames:
                    with open(fname, 'rb') as infile:  # open in binary mode also
                        outfile.write(infile.read())
            with py7zr.SevenZipFile(tmp_path, mode='r') as z:
                z.extractall(path=root)
            print(f"Extracted {archive_path} to {root}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if train:
            suf='train'
        else:
            suf='test'
        self.data = np.load(folder_path+'X_'+suf+'.npy')
        self.targets = np.load(folder_path+'y_'+suf+'.npy')
        self.transform = transform
        self.data = torch.unsqueeze(torch.Tensor(self.data), dim=1)
        self.data = torch.einsum('bxyz->bzxy', self.data)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class HarDataset(Dataset):
    def __init__(self, root='data', train=True, transform=None):
        folder_path=root+'/HAR/'
        archive_path = root+'/HAR.7z'

        if not os.path.exists(folder_path):
            print(f"Extracting {archive_path} to {root}...")
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                z.extractall(path=root)
            print(f"Extracted {archive_path} to {root}")

        if train:
            suf='train'
        else:
            suf='test'
        self.data = np.load(folder_path+'X_'+suf+'.npy')
        self.targets = np.load(folder_path+'y_'+suf+'.npy')
        self.transform = transform
        self.data = torch.Tensor(self.data)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


if __name__ == '__main__':
    dt=PamapDataset(root='../../data')