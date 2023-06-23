# FLDPCR-AuBCR
# Code for "Boosting Accuracy of Differential Privacy Continuous Data Release for Federated Learning"

The project is the code for the paper **"Boosting Accuracy of Differential Privacy Continuous Data Release for Federated Learning"**.

In this project, we design a novel differential privacy continuous data release (DPCR) model called augmented BIT-based continuous data release (AuBCR) model.

Thanks to the open sourced [**FLDPCR framework**](https://github.com/imcjp/FLDPCR). Our work is conducted based on the [**FLDPCR framework**]. By introducing the augmented policy matrix, designing the BIT-based DPCR algorithm with consistency (BCRC), and proposing the meta-factor approach, our AuBCR effectively boosts the learning accuracy compared to the state-of-the-art DPCR models.

The software environment of the source code requires **Python3**, and parts of the required packages are as follows:
* numpy
* scipy
* **opacus==1.1.2**
* torch==1.11.0+cu113
* torchaudio==0.11.0+cu113
* torchvision==0.12.0+cu113
* tensorboard==2.10.1

#### Instructions:

1. Our proposed AuBCR model is implemented in the file [**aubcr.py**](dpcrpy/bitMethods/aubcr.py).
2. The design of AuBCR is based on a rigorous theoretical derivation. Limited by the length of the paper, we show the relevant theoretical derivations in [**poof.pdf**](poof.pdf).
3. FL-DPCR framework is implemented in the folder "fldpcr", including server for the collaborator (server.py) and client for the participants (priClient.py for private FL and flClient.py for non-private FL), as well as some auxiliary codes in folder "utils".
4. We provide a script "main.m" to help users quickly implement FL-DPCR using our code. In this script, we provide multiple DPCR models, including SimpleMech, TwoLevel, BinMech, FDA, BCRG, ABCRG as well as our proposed AuBCRComp (note: AuBCR means $\left( k,N \right)$-AuBCR Model that supports $2^k$ releases).
5. The folder "dpcrpy" and "opacus_dpcr" are for implementing DPCR and private learning with DPCR. They are from [**Opacus-DPCR**](https://github.com/imcjp/Opacus-DPCR). Also, you can install our achievements with "pip" command as follows:
+ pip install **dpcrpy**
+ pip install **opacus_dpcr**
