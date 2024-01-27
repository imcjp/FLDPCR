# Code for "A Federated Learning Framework Based on Differentially Private Continuous Data Release"

The project is the code for the paper **"A Federated Learning Framework Based on Differentially Private Continuous Data Release"**. The authors are Jianping Cai, Ximeng Liu*, Qingqing Ye, Yang Liu, Yuyang Wang.

In this project, we design a Federated learning (FL) framework FL-DPCR. The framework introduces differential privacy continuous data release (DPCR) to construct parameter models, which significantly reduces the accumulation of noise on parameter models, thus improves the learning accuracy. By Equivalent Aggregation Theorem, we demonstrate that with proper algorithm design, traditional federated average aggregation can be transformed equivalently into the form of each participant individually constructing their local models before aggregation. Meanwhile, it addresses the effectiveness problem of DPCR applied to FL-DPCR and shows that the release error of DPCR is strongly associated with the error of the global model in the learning process of FL-DPCR. We verified the theorem in [**"Validation Report for Equivalent Aggregation Theorem"**](https://github.com/imcjp/FLDPCR/blob/main/Validation%20Report%20for%20Equivalent%20Aggregation%20Theorem.pdf).

The project integrates our designed [**Opacus-DPCR**](https://github.com/imcjp/Opacus-DPCR) to implement FL with DPCR. For more details, please see our [**Github Project**](https://github.com/imcjp/Opacus-DPCR).

The software environment of the source code requires **Python3**, and parts of the required packages are as follows:
* torch==2.1.2+cu118
* torchvision==0.16.2+cu118
* torchaudio==2.1.2+cu118
* **opacus==1.1.2**
* **opacus-dpcr==0.1.2**
* tensorboard==2.10.1
* py7zr

**Note: This project is designed based on Opacus version 1.1.2. Due to significant interface differences between versions of opacus, if you have installed a different version of Opacus, please reinstall Opacus version 1.1.2. Thank you!**


#### Instructions:

1. In folder "fldpcr", we implemented FL-DPCR, including server for the collaborator (server.py) and client for the participants (priClient.py for private FL and flClient.py for non-private FL), as well as some auxiliary codes in folder "utils".
2. We provide a script "main.m" to help users quickly implement FL-DPCR using our code. In this script, we provide multiple DPCR models, including SimpleMech, TwoLevel, BinMech, FDA as well as our proposed BCRG and ABCRG.
3. The project is based on [**Opacus-DPCR**](https://github.com/imcjp/Opacus-DPCR). You can install our achievements with "pip" command as follows:
+ pip install **dpcrpy**
+ pip install **opacus_dpcr**
4. We conduct our experiments on three datasets: **MNIST**, **FashionMNIST**(**FMNIST**) and **CIFAR10**. Additionally, we integrated two deep learning models **CNN** and **WideResNet**. You can use them with the following three commands:
+ For **MNIST**:
```bash
python main.py "{'data_config':{'dataset_name':'MNIST'}, 'model_config':{'name':'CNN_MNIST'}, 'dp_config':{'epsilon':1, 'delta':0.000333}, 'dpcr_model':{'name':'ABCRG'}}"
```
+ For **FashionMNIST**:
```bash
python main.py "{'data_config':{'dataset_name':'FashionMNIST'}, 'model_config':{'name':'CNN_MNIST'}, 'dp_config':{'epsilon':2, 'delta':0.000333}, 'dpcr_model':{'name':'ABCRG'}}"
```
+ For **CIFAR10**:
```bash
python main.py "{'data_config':{'dataset_name':'CIFAR10'}, 'model_config':{'name':'WideResnet10_2'}, 'dp_config':{'epsilon':8, 'delta':0.0004}, 'dpcr_model':{'name':'ABCRG'}}"
```
+ For **HAR**:
```bash
python main.py "{'data_config':{'dataset_name':'HAR'}, 'model_config':{'name':'CNN_HAR'}, 'dp_config':{'epsilon':4, 'delta':0.000625}, 'fed_config':{'K':5}, 'dpcr_model':{'name':'ABCRG'}}"
```
+ For **PAMAP**:
```bash
python main.py "{'data_config':{'dataset_name':'PAMAP'}, 'model_config':{'name':'CNN_PAMAP'}, 'dp_config':{'epsilon':4, 'delta':0.000357}, 'fed_config':{'K':5}, 'dpcr_model':{'name':'ABCRG'}}"
```

If you have any questions or suggestions for improvement, please contact email jpingcai@163.com. Thank you!

