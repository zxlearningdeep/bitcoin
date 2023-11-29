import numpy as np
import torch
import torch as th
from torch import Tensor
import warnings


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils.load_data import load_data
from utils.params import set_params
from module.preprocess import remove_self_loop
warnings.filterwarnings('ignore')

EPS = 1e-15


homo_ratio = torch.load('./data/t_fin/hr.pt').cpu()

# yelp 1d Draw Plot
plt.figure(figsize=(10, 7), dpi=180)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams.update({'font.size':15})
plt.title('Density of Homophily Ratio (1D)', fontsize=25, fontweight='bold')

# sns.kdeplot(homo_ratio[0].numpy(), fill=True, color="g", label="homo ratio", alpha=.7)
sns.kdeplot(homo_ratio.numpy(), fill=True, color="y", alpha=0.8)

# darkseagreen
plt.legend(loc='upper right')

plt.xlim(0,1)
# plt.ylim(0,1)

plt.xlabel('Node-Level Homophily Ratio',fontsize=20,fontweight='bold')
plt.ylabel('Density',fontsize=20,fontweight='bold')
plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')

plt.savefig('./fig/t_fin_hr.svg', dpi=600, format='svg')
plt.show()

