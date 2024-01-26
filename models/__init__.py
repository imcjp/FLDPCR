from .mnist import *
from .wideresnet import *
from .iotModels import *
def gen(name,args=None):
    if args is None:
        return eval(name)();
    else:
        return eval(name)(**args);
