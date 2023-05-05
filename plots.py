import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class get_images:
    def __init__(self, path='results/cicids/all_features/conf_matrix_basearch_all_full_balanced.pickle'):
        self.accrd, self.ataccrd=self.get_dataframes(path)
    
    def get_dataframes(self,path):
        
        with open(path, 'rb') as file:
            cfs = pickle.load(file)
    
        accuracy = []
        attack_accuracy = []

        for i in range(len(cfs)):
            accuracy.append(np.round(np.trace(cfs[i])/np.sum(cfs[i])*100,2))
            attack_accuracy.append(np.nan_to_num(np.round(cfs[i][1][1]/(cfs[i][0][1]+cfs[i][1][1])*100,2)))
    
        attacks = [1,2,3,4,5,6,7,10,11,12,14]# Attacks that are retained
        # Convert from 1D to 2D
        numattacks = len(attacks)
        accr = np.zeros((numattacks, numattacks))
        attackaccr = np.zeros((numattacks, numattacks))

        n = 0

        for i in range(numattacks):
            for j in range(numattacks):
                if i == j:
                    accr[i][j] = 100
                    continue
                else:
                    accr[i][j] = accuracy[n]
                    attackaccr[i][j] = attack_accuracy[n]
                    n = n + 1
        
        # Create Datafeame and update row/column names
        accrd = pd.DataFrame(accr)
        ataccrd = pd.DataFrame(attackaccr)

        #accrd.columns=accrmat.columns.values
        accrd.columns=attacks
        accrd.index=attacks
        ataccrd.columns=attacks
        ataccrd.index=attacks
        
        return accrd, ataccrd
    
    def get_overall_accuracy(self):


        fig = plt.figure(figsize=(10, 8), dpi=150)

        plt.rcParams["font.family"] = "serif"
        plt.rc('axes', titlesize=12)     # fontsize of the axes title
        plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
        plt.rc('legend', fontsize=11)    # legend fontsize
        plt.rc('axes', titlesize=15)  # fontsize of the figure title
        # create some labels
        attack_labels = [ "{0:d}".format(i) for i in attacks ]

        ax = fig.add_subplot(111)
        sns.heatmap(self.accrd, linewidths=.5, annot=True, fmt='.2f', cmap='winter', ax=ax)


        plt.xlabel('Test Attack')
        plt.ylabel('Training Attack')
        plt.title('Overall accuracy when training and testing with individual attacks')
        
        return plt
        
    def get_attack_accuracy(self):


        fig = plt.figure(figsize=(10, 8), dpi=150)

        plt.rcParams["font.family"] = "serif"
        plt.rc('axes', titlesize=12)     # fontsize of the axes title
        plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
        plt.rc('legend', fontsize=11)    # legend fontsize
        plt.rc('axes', titlesize=15)  # fontsize of the figure title
        # create some labels
        attack_labels = [ "{0:d}".format(i) for i in attacks ]

        ax = fig.add_subplot(111)
        sns.heatmap(self.ataccrd, linewidths=.5, annot=True, fmt='.2f', cmap='winter', ax=ax)


        plt.xlabel('Test Attack')
        plt.ylabel('Training Attack')
        plt.title('Attack Accuracy when training and testing with individual attacks (Second Run)')

        return plt