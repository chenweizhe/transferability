import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from model import cldnn
from plots import get_images
from dataloader import create_dataloader
import numpy as np
import math
import os
import random
import pickle
import tensorflow as tf
import argparse
#which gpu is available?
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

path='results/cicids/all_features/'

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--bootstrap',type=str,default='False')

#block to get reproducible results
seed_value=1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

accs = []
cfs = []

dataloader=create_dataloader()
y_attacks=dataloader.get_y_attacks()

bootstrap=False if args.bootstrap=='False' else True

for n_attacks_train in y_attacks:
        
    cldnn = cldnn()
    cldnn.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    
    X_train,Y_train=dataloader.get_train_data(n_attacks_train,bootstrap)
        
    cldnn_mdata=cldnn.fit(X_train,Y_train,validation_split=0.2,epochs=20,batch_size =1024,shuffle=True)
    
    if not os.path.exists(path+f'weights/{n_attacks_train}'):
        os.makedirs(path+f'weights/{n_attacks_train}')
    
    cldnn.save(path+f'weights/{n_attacks_train}/weights.h5')
    
    for n_attacks_test in y_attacks:
        #print(n_attacks_train, n_attacks_test)

        #find the testing attacks
        if n_attacks_test== n_attacks_train:
            continue
        
        X_test, Y_test=dataloader.get_test_data(n_attacks_test)

        scores=cldnn.evaluate(X_test,Y_test)
        
        print(n_attacks_train, n_attacks_test)
        print("Accuracy:%.2f%%"%(scores[1]*100))
        
        accs.append(scores[1])
        
        y_pred=cldnn.predict(X_test)
        y_label=np.argmax(y_pred,axis=-1)
        cf=confusion_matrix(y_test,y_label)
        
        cfs.append(cf)
        
# Name of pickle file to be saved
outname = 'conf_matrix_basearch_all_full_balanced.pickle'

#outfile = os.path.join(os.getcwd(), outname)
if os.path.exists(path+outname):
        os.replace(path+outname, path+outname + ".old")
        
with open(path+outname, 'wb') as file:
    pickle.dump(cfs, file)
    
img=get_image()
plt=img.get_overall_accuracy()
plt.show()