import os
import os
from sklearn.model_selection import train_test_split
import shutil

source = 'dataset/D4'
dest = 'dataset/d4new/'

os.makedirs(dest+'train/0')
os.makedirs(dest+'train/1')
os.makedirs(dest+'val/0')
os.makedirs(dest+'val/1')


path0_train = dest + 'train/0/'
path1_train = dest + 'train/1/'
path0_val = dest + 'val/0/'
path1_val = dest + 'val/1/'

x = []
y = []
for f in sorted(os.listdir(source)):
    if os.path.isdir(os.path.join(source,f)):
        for i in sorted(os.listdir(os.path.join(source,f))):
            x.append(os.path.join(source, f, i))
            y.append(f)

X_train, X_val, label_train, label_val = train_test_split(x, y,  train_size=0.7)
for data, label in zip(X_train,label_train):
    print(data, label)
    if (label == '0'):
        shutil.copy(os.path.join(data), path0_train)
    else:
        shutil.copy(os.path.join(data), path1_train)
for data, label in zip(X_val,label_val):
    print(data, label)
    if (label == '0'):
        shutil.copy(os.path.join(data), path0_val)
    else:
        shutil.copy(os.path.join(data), path1_val)
