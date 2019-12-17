
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from pandas.plotting import scatter_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
import math 

# ##################  plot_confusion_matrix function declaration ########################################
def plot_confusion_matrix(y_true,y_pred):
    
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Count ', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    
    if flag_save:
        plt.savefig("Confusion_matrix.png")
    else:
        plt.show()
#////////////////////////////////////////////////////////////////////////////////////////
def main():
    try:
        # read the data
        df = pd.read_csv("f_10k.csv")
        
        # get the inputting data
        input_dt = df.ix[:, 2:-1]
        # get the output data
        output_dt = df.ix[:, -1]

        print(df.head(20))
        print(df.describe())
        # ********************************************************************
        # index for string type data
        index = [1,2,5,6]
        # index for numerical type data
        index1 = [0, 3, 4, 7,8,9,10]

        # To convert data of inputting string type into numerical type*******************
        for i in index:
            # get the colmumn data
            col_dt = input_dt.ix[:, i]
            # create the encoder to convert string type into numerical data
            encoder = preprocessing.LabelEncoder()
            # Fit the encoder to convert string type into numerical data
            encoder.fit(col_dt)
            # Convert the encoder
            encoded_values = encoder.transform(col_dt)
            # put the original data into encoded data
            input_dt.ix[:, i] = encoded_values

        # To normalize data of inputing value type *******************
        for i in index1:
            #get the colmumn data
            col_dt = np.array(input_dt.ix[:, i])
            # convert the 1D data into 2D to calculate the result
            col_dt1 =[[x] for x in col_dt]
            # create the encoder to normalize data
            data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            # implement the result
            data_scaled = data_scaler.fit_transform(col_dt1)
            input_dt.ix[:, i] = data_scaled
        for i in index:
            #get the colmumn data
            col_dt = np.array(input_dt.ix[:, i])
            # convert the 1D data into 2D to calculate the result
            col_dt1 =[[x] for x in col_dt]
            # create the encoder to normalize data
            data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            # implement the result
            data_scaled = data_scaler.fit_transform(col_dt1)
            input_dt.ix[:, i] = data_scaled  

        #********************************************************************
        print("-------------- modified input data -------------------")
        print(input_dt.head(20))

        # To convert output data into value type to find the result *******************
        # create the encoder to convert string type into numerical data
        encoder = preprocessing.LabelEncoder()
        # Fit the encoder to convert string type into numerical data
        encoder.fit(output_dt.loc[:])
        # Convert the encoder
        encoded_values = encoder.transform(output_dt)
        # put the original data into encoded data
        output_dt.loc[:] = encoded_values

        print("-------------- modified Label -------------------")
        print(output_dt.head(20))

        # show the distribution of the input attributes ****************
        figsize = (12, 12)       
        title = 'Distribution of the input attributes'
        input_dt.plot(kind='box', subplots=True, layout=(3, 4), sharex=False, sharey=False, title=title, figsize=figsize)    
        if flag_save:
            plt.savefig("dist_input.png")
        else:
            plt.show()      
        
        # show the Correlation of the input attributes ****************    
        colormap = plt.cm.viridis
        plt.figure(figsize=(12, 12))
        input_tmp = input_dt.copy()
        input_tmp = input_tmp.drop(['forwarding status'], axis=1)
        title = 'Correlation of the input attributes'
        plt.title(title)
        sns.heatmap(input_tmp.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, annot=True)
        if flag_save:
            plt.savefig("corr_input.png")
        else:
            plt.show()      
                 
        # Scatter Plots of the input attributes & Output **************** 
        input_tmp['output'] = output_dt   
        fig = plt.figure(figsize=(20, 20))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        sns.set(font_scale=1.4)
        for i in range(1, len(input_tmp.columns)):
            ax = fig.add_subplot(3, 4, i)          
            if i== 8:
                ax = sns.scatterplot(x=input_tmp.columns[i-1], y="output", hue="output", style="output", data=input_tmp, ax=ax, s=200) 
                ax.legend(bbox_to_anchor=(1.05, 0.3), loc='lower left', borderaxespad=0.)
            else:
                ax = sns.scatterplot(x=input_tmp.columns[i-1], y="output", hue="output", style="output", data=input_tmp, ax=ax, legend=False, s=200)                                
       
        title = 'Scatter Plots of the input attributes & Output'            
        plt.suptitle(title, y=0.95)           
        if flag_save:
            plt.savefig("scatter_input_output.png")
        else:
            plt.show()      

        
        # show the histogram of the input attributes ***************
        figsize = (16, 16)       
        title = 'Histogram of the input attributes'
        g = input_dt.hist(figsize=figsize)        
        plt.suptitle(title, y=0.95)
        if flag_save:
            plt.savefig("hist_input.png")
        else:
            plt.show()      


         # show the histogram of the output attributes ***************
        title = 'Histogram of the output attributes'
        fig = plt.figure(figsize=(20, 20))
        output_dt.hist()
        plt.suptitle(title, y=0.95)
        if flag_save:
            plt.savefig("hist_output.png")
        else:
            plt.show()      


        #****** to split the data into taining and testing data to train the a neural network **************
        # convert inputing and outputing data to array type to estimate the result
        input_dt = np.array(input_dt)
        output_dt = np.array(output_dt)

        # split the data into taining and testing data
        trainX, testX, trainY, testY = train_test_split(input_dt,output_dt,test_size = 0.40, random_state = 42)

        # *** To create and train a network**********************
        # to create Multi-layer neural network
        clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(5,), random_state=1)

        # to train Multi-layer neural network with training data
        clf_MLP.fit(trainX, trainY)

        # to find the output by Multi-layer neural network with test data
        y_pred_MLP = clf_MLP.predict(testX)
        
        
        
        # *** To find the confusion  ****************************************

        # find confusion matrix on testing data
        confusion = confusion_matrix(testY, y_pred_MLP)
        print("----- confusion matrix ----------")
        print(confusion)

        # show confusion matrix on testing data as the graph
        sns.reset_orig()
        plot_confusion_matrix(testY, y_pred_MLP)
        sns.set()

        # *** To find another parameter for accuracy dicision *********

        print('--------- Parameters-------------------------')
    

        # To find the Precision value
        Precision = confusion[1, 1] / sum(confusion[ :,1])
        print('Precision  is %0.4f' % Precision)

        # To find the Sensitivity value
        #Sensitivity = confusion[0, 1] / sum(confusion[:, 1])
        #print('Sensitivity  is %0.4f' % Sensitivity)

        # To find the Recall value
        Recall = confusion[1, 1] / sum(confusion[1,:])
        print('Specificity  is %0.4f' % Recall)

        # To find the Specificity value
        #print('Recall  is %0.4f' % Sensitivity)

        # To find the F_score value
        F_score = 2 * Precision * Recall / (Precision + Recall)
        print('F_score  is %0.4f' % F_score)

        # To find the Accuracy value
        Accuracy = (confusion[1, 1] + confusion[0, 0]) / (confusion[1, 1] + confusion[0, 0]+ confusion[1, 0] + confusion[0, 1])
        print('Accuracy  is %0.4f'% Accuracy)
        # TO find Matthews correlation coefficient
        MCC = ((confusion[0, 0] * confusion[1, 1])-(confusion[0, 1])*(confusion[1, 0])) / (math.sqrt((confusion[0, 0]+confusion[1, 0])*(confusion[0, 0]+confusion[0, 1]) *(confusion[1, 1]+confusion[1, 0]) *(confusion[1, 1]+confusion[0, 1])))
        print ('MCC  is %0.4f'% MCC)
        
        
        probs = clf_MLP.predict_proba(testX)
        probs = probs[:, 1]
    
        auc_roc = roc_auc_score(testY, probs)
        print('AUC ROC: %.3f' % auc_roc)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(testY, probs)
        # plot no skill
        fig = plt.figure(figsize=(12, 12))
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.')
        # show the plot
        plt.title('ROC Curve')
        plt.plot(fpr, tpr, label='AUC = %0.4f'% auc_roc)
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        if flag_save:
            plt.savefig("roc_curve.png")
        else:
            plt.show()      
        

        # calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(testY, probs)
        # calculate F1 score
        f1 = f1_score(testY,y_pred_MLP)
        # calculate precision-recall AUC
        auc_recall = auc(recall, precision)
        # calculate average precision score
        ap = average_precision_score(testY, probs)
        #print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_recall, ap))
        print('AUC RECALL: %.3f' % auc_recall)
        # plot no skill
        fig = plt.figure(figsize=(12, 12))
        plt.plot([0, 1], [0.5, 0.5], linestyle='--')
        # plot the precision-recall curve for the model
        plt.plot(recall, precision, marker='.', label='AUC = %0.4f'% auc_recall)
        # show the plot
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower right')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        if flag_save:
            plt.savefig("Precision-Recall.png")
        else:
            plt.show()      

    except ValueError:
        print(ValueError)



##################################################
if __name__=="__main__":
    flag_save = True
    main()
    


# In[2]:


#df.describe()


# In[3]:



