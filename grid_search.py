import os
import re
from My_network import My_network
from main import My_dataset
from main import MyTransform
import tqdm
import numpy as np
import json
file_lst=[]
transform = MyTransform()
test_dataset = My_dataset(root='./data', train=False, transform=transform)
X_test, Y_test = [], []
for i in tqdm.tqdm(range(len(test_dataset))):
    x, y = test_dataset[i]
    X_test.append(x)
    Y_test.append(y)
X_test = np.array(X_test)
Y_test_onehot = np.zeros((len(test_dataset), 10))
for i, y in enumerate(Y_test):
    Y_test_onehot[i, y] = 1
for files in os.listdir('.'):
    if files.endswith('params.npz'):
        file_lst.append(files)
pattern= r"L2_([0-9.]+)_lr_([0-9.]+)_layers_([0-9]+)_([0-9]+)_params\.npz"
params_result=[]

for file in file_lst:
    match=re.match(pattern,file)
    if match:
        l2=float(match.group(1))
        lr=float(match.group(2))
        layers_1=int(match.group(3))
        layers_2=int(match.group(4))
        model=My_network(3072,layers_1,layers_2,10,L2_lamda=l2,activa_function='Relu',learning_rate=lr)
        model.load_params(file)
        acc=model.evaluate(X_test,Y_test_onehot)
        params_result.append({'L2':l2,'lr':lr,'layer1':layers_1,'layer2':layers_2,'accuracy':acc})
        print(acc)
with open('params_result.json','w') as f:
    json.dump(params_result,f,indent=4)



