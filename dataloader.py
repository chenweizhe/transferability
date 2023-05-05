import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np

class create_dataloader:
    def __init__(self, root_path='../MachineLearningCVE/'):
        #import dataset
        self.dataset=pd.read_csv(root_path+'Tuesday-WorkingHours.pcap_ISCX.csv')
        dataset_1=pd.read_csv(root_path+'Monday-WorkingHours.pcap_ISCX.csv')
        dataset_2=pd.read_csv(root_path+'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
        dataset_3=pd.read_csv(root_path+'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
        dataset_4=pd.read_csv(root_path+'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
        dataset_5=pd.read_csv(root_path+'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
        dataset_6=pd.read_csv(root_path+'Wednesday-workingHours.pcap_ISCX.csv')
        dataset_7=pd.read_csv(root_path+'Friday-WorkingHours-Morning.pcap_ISCX.csv')

        self.dataset=self.dataset.append(dataset_1, ignore_index=True)
        self.dataset=self.dataset.append(dataset_2, ignore_index=True)
        self.dataset=self.dataset.append(dataset_3, ignore_index=True)
        self.dataset=self.dataset.append(dataset_4, ignore_index=True)
        self.dataset=self.dataset.append(dataset_5, ignore_index=True)
        self.dataset=self.dataset.append(dataset_6, ignore_index=True)
        self.dataset=self.dataset.append(dataset_7, ignore_index=True)
        
        self.clean()
        self.x_dataset_scaled, self.y_encoded=self.get_data()
        self.x_normal_train, self.x_normal_test, self.y_normal_train,self.y_normal_test=self.get_benign_indices()
                 
    def clean(self):
        self.dataset[self.dataset==np.inf]=np.nan
        self.dataset.fillna(0,inplace=True)
        return 
   
    def get_data(self):             
        x_dataset=self.dataset.iloc[:, :-1]
        y=self.dataset.iloc[:, -1]
        
        le=LabelEncoder()
        y_encoded=le.fit_transform(y)
        
        scaler=MinMaxScaler()
        x_dataset_scaled=scaler.fit_transform(x_dataset)
        
        return x_dataset_scaled,y_encoded
          
    def to_onehot(self,yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(yy)),yy] = 1
        return yy1
    
    def get_benign_indices(self):
                 
        #FIND OUT INDICES OF Y AND X THAT ARE ONLY NORMAL AND DIVIDE intow two parts
        y_normal_indices=np.where(self.y_encoded==0)
        y_normal_indices_list=np.array_split(y_normal_indices[0],2)
        y_normal_train=self.y_encoded[y_normal_indices_list[0]]
        y_normal_test=self.y_encoded[y_normal_indices_list[1]]
        x_normal_train=self.x_dataset_scaled[y_normal_indices_list[0]]
        x_normal_test=self.x_dataset_scaled[y_normal_indices_list[1]]
        
        return x_normal_train, x_normal_test, y_normal_train, y_normal_test
                 
    def get_train_data(self, n_attacks_train, bootstrap=False):
        #find the training attacks
        y_train_indices=np.where(self.y_encoded==n_attacks_train)

        y_attack_train=np.ones(len(y_train_indices[0]),dtype=int)  
        #find the x_dataset corr to this
        x_attack_train=self.x_dataset_scaled[y_train_indices[0]]
    
        #create the training data
        x_train=np.concatenate([x_attack_train,self.x_normal_train])
        y_train=np.concatenate([y_attack_train,self.y_normal_train])
    
        print("Length of y_attack_test",len(y_attack_train))
        n=(int)(len(self.y_normal_train)/len(y_attack_train))
        print(n)
        # X_train, Y_train=get_train_data(n_attacks_train, bootstrap)
        
        if bootstrap:
            for i in range(n-1):
                x_train=np.concatenate([x_train,x_attack_train])
                y_train=np.concatenate([y_train,y_attack_train])
        print('TRAINING',len(y_train), len(y_train[y_train==0]), len(y_train[y_train==1]))
        
        x_train_examples,_=x_train.shape
        Y_train = self.to_onehot(y_train)
        X_train = x_train.reshape(x_train_examples, 2, 39, 1)

        return X_train, Y_train
    
    def get_test_data(self, n_attacks_test):
        
        y_test_indices=np.where(self.y_encoded==n_attacks_test)
        y_test_indices=y_test_indices[0] 
        
        y_attack_test=np.ones(len(y_test_indices),dtype=int)
        
        #print(len(x_train),len(y_train[y_train==0]),len(y_train[y_train==1]))
            
        #find the x_dataset corr to this
        x_attack_test=self.x_dataset_scaled[y_test_indices]

        #create the testing data
        x_test=np.concatenate([x_attack_test,self.x_normal_test])
        y_test=np.concatenate([y_attack_test,self.y_normal_test])   
    
        #print("TRAINING", len(y_train),len(y_train[y_train==0]), len(y_train[y_train==1]))
        print("TESTING", len(y_test),len(y_test[y_test==0]), len(y_test[y_test==1]))   
        
        x_test_examples,_= x_test.shape
                
        
        Y_test = self.to_onehot(y_test)
        X_test = x_test.reshape(x_test_examples, 2, 39, 1)
        
        return X_test,Y_test
         
    
    def get_y_attacks(self):

        y_val,y_count = np.unique(self.y_encoded, return_counts=True) 
        y_retained = y_val[y_count>100]
        y_attacks = y_retained[y_retained>0]
        
        return y_attacks
    
    def temporal_average(data, labels):
        ind_attack=np.where(labels==1)[0]
        ind_benign=np.where(labels==0)[0]
    
        data_att=data[ind_attack]
        data_ben=data[ind_benign]
    
        label_att=labels[ind_attack]
        label_ben=labels[ind_benign]
    
        attack=get_avgd_data(data_att)
        benign=get_avgd_data(data_ben)
    
        attack=np.concatenate([attack,benign])
        label_att=np.concatenate([label_att, label_ben])
    
        return attack, label_att

    def get_avgd_data(data):
        n=3 if len(data)>3 else len(data)
    
        data_avg=np.zeros(data.shape)

        for i in range(len(data)):
            for j in range(i,i+n):
                data_avg[i,:]+=data[j,:] if j<len(data) else data[j-len(data)]
            data_avg[i,:]/=n
        return data_avg