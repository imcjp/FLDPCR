# Code for "A Federated Learning Framework Based on Differential Privacy Continuous Data Release"

The project is the code for the paper **"A Federated Learning Framework Based on Differential Privacy Continuous Data Release"**. The authors are Jianping Cai, Ximeng Liu*

In this project, we design a Federated learning (FL) framework FL-DPCR. The framework introduces differential privacy continuous data release (DPCR) to construct parameter models, which significantly reduces the accumulation of noise on parameter models, thus improves the learning accuracy.

The project integrates our designed [**Opacus-DPCR**](https://github.com/imcjp/Opacus-DPCR) to implement FL with DPCR. For more details, please see our [**Github Project**](https://github.com/imcjp/Opacus-DPCR).

The software environment of the source code requires **Python3**, and parts of the required packages are as follows:
* numpy
* scipy
* opacus==1.1.2
* torch==1.11.0+cu113
* torchaudio==0.11.0+cu113
* torchvision==0.12.0+cu113
* tensorboard==2.10.1

#### Instructions:

1. In folder "fldpcr", we implemented FL-DPCR, including server for the collaborator (server.py) and client for the participants (priClient.py for private FL and flClient.py for non-private FL), as well as some auxiliary codes in folder "utils".
2. We provide a script "main.m" to help users quickly implement FL-DPCR using our code. In this script, we provide multiple DPCR models, including SimpleMech, TwoLevel, BinMech, FDA as well as our proposed BCRG and ABCRG.
3. The folder "dpcrpy" and "opacus_dpcr" are for implementing DPCR and private learning with DPCR. They are from [**Opacus-DPCR**](https://github.com/imcjp/Opacus-DPCR). Also, you can install our achievements with "pip" command as follows:
+ pip install **dpcrpy**
+ pip install **opacus_dpcr**

If you have any questions or suggestions for improvement, please contact email jpingcai@163.com. Thank you!

