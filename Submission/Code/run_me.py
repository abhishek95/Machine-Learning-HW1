import numpy as np
import scipy
import sklearn
import matplotlib
from sklearn.neighbors import KNeighborsRegressor
from math import floor
import math
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso,Ridge
from math import pow
import time
import matplotlib.pyplot as plt
from sklearn import tree


# Import python modules
import kaggle

# Read in train and test data
def read_data_power_plant():
	print('Reading power plant dataset ...')
	train_x = np.loadtxt('../../Data/PowerOutput/data_train.txt')
	train_y = np.loadtxt('../../Data/PowerOutput/labels_train.txt')
	test_x = np.loadtxt('../../Data/PowerOutput/data_test.txt')
	return (train_x, train_y, test_x)

def read_data_localization_indoors():
	print('Reading indoor localization dataset ...')
	train_x = np.loadtxt('../../Data/IndoorLocalization/data_train.txt')
	train_y = np.loadtxt('../../Data/IndoorLocalization/labels_train.txt')
	test_x = np.loadtxt('../../Data/IndoorLocalization/data_test.txt')
	return (train_x, train_y, test_x)
# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

############################################################################

#global variables

min_e=10000000
alpha_for_min_e=3
error_mat=[]
flag=True
time_mat=[]
min_n=3
time_mat=[]
depth_for_min_e=3

############################################################################ LINEAR REGRESSION


def crossval_kfold_Lasso(k,alpha,train_x,train_y):
    global error_mat
    global alpha_for_min_e
    global min_e
    time_taken=[]
    kf = KFold(n_splits=k,shuffle=True)
    kf = kf.split(train_x,train_y)
    e=[]
    
    for train in kf:
        training_x=[]
        training_y=[]
        testing_x =[]
        testing_y=[]
        start_time=time.time()
        train_indices = train[0]
        test_indices = train[1]
        for idx in train_indices:
            training_x.append(train_x[idx])
            training_y.append(train_y[idx])
        for idx in test_indices:
            testing_x.append(train_x[idx])
            testing_y.append(train_y[idx]) 
            
        clf = Lasso(alpha=alpha,copy_X=True,max_iter=15)
        clf = clf.fit(training_x,training_y)
        y_hat = clf.predict(testing_x)
        e.append(compute_error(y_hat,testing_y))
        
        end_time=time.time()
        delta_time = (end_time-start_time)*1000
        time_taken.append(delta_time)
        del (clf,training_x,training_y,testing_x,testing_y)
    
    avg_time = sum(time_taken)/float(len(time_taken))
    time_taken.append(avg_time)
    time_mat.append(time_taken)
    
    print (str(e))
    avg_e = sum(e)/float(len(e))
    e.append(avg_e)
    error_mat.append(e)
    
    min_e = min(min_e,avg_e)
    if(min_e==avg_e): alpha_for_min_e = alpha
        

def crossval_kfold_Ridge(k,alpha,train_x,train_y):
    global error_mat
    global alpha_for_min_e
    global min_e
    global flag
    time_taken=[]
    kf = KFold(n_splits=k,shuffle=True)
    kf = kf.split(train_x,train_y)
    e=[]
    for train in kf:
        training_x=[]
        training_y=[]
        testing_x =[]
        testing_y=[]
        start_time=time.time()
        train_indices = train[0]
        test_indices = train[1]
        for idx in train_indices:
            training_x.append(train_x[idx])
            training_y.append(train_y[idx])
        for idx in test_indices:
            testing_x.append(train_x[idx])
            testing_y.append(train_y[idx]) 
            
        clf = Ridge(alpha=alpha,copy_X=True,max_iter=15)
        clf = clf.fit(training_x,training_y)
        y_hat = clf.predict(testing_x)
        e.append(compute_error(y_hat,testing_y))
        end_time=time.time()
        delta_time = (end_time-start_time)*1000
        time_taken.append(delta_time)
        del (clf,training_x,training_y,testing_x,testing_y)
    
    avg_time = sum(time_taken)/float(len(time_taken))
    time_taken.append(avg_time)
    time_mat.append(time_taken)
    
   
    avg_e = sum(e)/float(len(e))
    e.append(avg_e)
    error_mat.append(e)
    print (str(e))
        
    min_e = min(min_e,avg_e)
    if(min_e==avg_e): 
        alpha_for_min_e = alpha
        flag=False


def choose_best_LinearModel(train_x,train_y,test_x,alphas,k_fold):
    global error_mat
    global alpha_for_min_e
    global min_e
    global time_mat
    error_mat[:]=[]
    min_e=10000000
    time_mat[:]=[]
    print (str(k_fold)+" fold cross validation:")
    for alpha in alphas:
        print ("Lasso regression training for alpha="+str(alpha)+":")
        crossval_kfold_Lasso(k_fold,alpha,train_x,train_y)
    #print (error_mat)
    
    for alpha in alphas:
        print ("Ridge regression training for alpha="+str(alpha)+":")
        crossval_kfold_Ridge(k_fold,alpha,train_x,train_y)
    #print (error_mat)
    avg_errors=[]
    for i in error_mat:
        avg_errors.append(i[k_fold])
    print ('Average errors:'+str(avg_errors))
    
    # retraining with full training data
    print ("re-training with full train data for alpha="+str(alpha_for_min_e))
    if (flag):
        clf = Lasso(alpha=alpha_for_min_e,copy_X=True)
        print('chosen Lasso model')
    else: 
        clf = Ridge(alpha=alpha_for_min_e,copy_X=True)
        print ('chosen Ridge model')
    clf = clf.fit(train_x,train_y)
    y_hat = clf.predict(train_x)
    e = compute_error(y_hat,train_y)
    print ("MAE train error="+str(e))
    y_pred = clf.predict(test_x)
    return y_pred



def indoor_localization_LinearModel(k_fold):
    train_x, train_y, test_x = read_data_localization_indoors()
    print('Train=', train_x.shape)
    print('Test=', test_x.shape)
    alphas = [pow(10,-6),pow(10,-4),pow(10,-2),1,10]
    y_pred = choose_best_LinearModel(train_x,train_y,test_x,alphas,k_fold)

    #####plot
    avg_time=[]
    log_alphas=[]
    for i in time_mat:
        avg_time.append(i[k_fold])
    for i in alphas:
        log_alphas.append(math.log10(i))
    plt.plot(log_alphas,avg_time[0:len(alphas)],'ro',label='Lasso')
    plt.plot(log_alphas,avg_time[len(alphas):2*len(alphas)],'bo',label='Ridge')
    plt.legend(loc='upper center')
    plt.ylabel('Avg Time for validation')
    plt.xlabel('Alpha for Linear Model (Log)')
    plt.title('Indoor Localization dataset (Model=Linear Model)')
    plt.show()
    
    
    predicted_y = y_pred
    file_name = '../Predictions/IndoorLocalization/indoor_localization_LM.csv'
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    
    file2 =open('ILlogfile_LM.txt','w')
    file2.write(str(error_mat)+'\n')
    file2.write(str(time_mat))
    file2.close()
    
    #clear global variables
    error_mat[:]=[]
    time_mat[:]=[]
    alpha_for_min_e=3
    flag=True
    min_e=100000000
    
    print('\n\n')

    
def power_plant_LinearModel(k_fold):
    train_x, train_y, test_x = read_data_power_plant()
    print('Train=', train_x.shape)
    print('Test=', test_x.shape) 
    alphas = [pow(10,-6),pow(10,-4),pow(10,-2),1,10]
    y_pred = choose_best_LinearModel(train_x,train_y,test_x,alphas,k_fold)
    
    #####plot
    avg_time=[]
    log_alphas=[]
    for i in time_mat:
        avg_time.append(i[k_fold])
    for i in alphas:
        log_alphas.append(math.log10(i))
    plt.plot(log_alphas,avg_time[0:len(alphas)],'ro',label='Lasso')
    plt.plot(log_alphas,avg_time[len(alphas):2*len(alphas)],'bo',label='Ridge')
    plt.legend(loc='upper center')
    plt.ylabel('Avg Time for validation')
    plt.xlabel('Alpha for Linear Model (Log)')
    plt.title('Power Plant dataset (Model=Linear Model)')
    plt.show()
    
    predicted_y = y_pred
    file_name = '../Predictions/PowerOutput/power_plant_LM.csv'
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    
    file2 =open('PPlogfile_LM.txt','w')
    file2.write(str(error_mat)+'\n\n\n')
    file2.write(str(time_mat))
    file2.close()
    
    #clear global variables
    error_mat[:]=[]
    time_mat[:]=[]
    alpha_for_min_e=3
    flag=True
    min_e=100000000
    avg_time[:]=[]
    log_alphas[:]=[]
    
    print('\n\n')


    
#####################################################################

############################################################################ KNN




    
def KNN(neighbors,train_x,train_y):
    neigh = KNeighborsRegressor(n_neighbors=neighbors)
    neigh = neigh.fit(train_x,train_y)
    return (neigh)

def crossval_kfold_KNN(k,neighbors,train_x,train_y):
    global error_mat
    global min_n
    global min_e
    time_taken=[]
    kf = KFold(n_splits=k,shuffle=True)
    kf = kf.split(train_x,train_y)
    training_x=[]
    training_y=[]
    testing_x =[]
    testing_y=[]
    e=[]
    for train in kf:
        start_time=time.time()
        train_indices = train[0]
        test_indices = train[1]
        for idx in train_indices:
            training_x.append(train_x[idx])
            training_y.append(train_y[idx])
        for idx in test_indices:
            testing_x.append(train_x[idx])
            testing_y.append(train_y[idx]) 
            
        neigh = KNeighborsRegressor(n_neighbors=neighbors)
        neigh = neigh.fit(training_x,training_y)
        y_hat = neigh.predict(testing_x)
        e.append(compute_error(y_hat,testing_y))
        end_time=time.time()
        delta_time = (end_time-start_time)*1000
        time_taken.append(delta_time)
    
    avg_time = sum(time_taken)/float(len(time_taken))
    time_taken.append(avg_time)
    time_mat.append(time_taken)
    
    print (str(e))
    avg_e = sum(e)/float(len(e))
    e.append(avg_e)
    error_mat.append(e)
    
    min_e = min(min_e,avg_e)
    if(min_e==avg_e): min_n = neighbors


def choose_best_KNN(train_x,train_y,test_x,k_fold,neighbour_list):
    global error_mat
    global min_n
    global min_e 
    print (str(k_fold)+" fold cross validation:")
    
    for neighbors in neighbour_list:
        print ("KNN regression training for n="+str(neighbors)+":")
        crossval_kfold_KNN(k_fold,neighbors,train_x,train_y)
    
    #print avg error from cross validations
    avg_errors=[]
    for i in error_mat:
        avg_errors.append(i[k_fold])
    print ('Average errors:'+str(avg_errors))
    
    # retraining with full training data
    print ("retraining with full train data for n="+str(min_n))
    neigh = KNeighborsRegressor(n_neighbors=min_n)
    neigh = neigh.fit(train_x,train_y)
    y_hat = neigh.predict(train_x)
    e = compute_error(y_hat,train_y)
    print ("MAE train error="+str(e))
    y_pred = neigh.predict(test_x)
    return y_pred
           


def power_plant_KNN():
    train_x, train_y, test_x = read_data_power_plant()
    print('Train=', train_x.shape)
    print('Test=', test_x.shape) 
    k_fold=5
    neighbour_list=[3,5,10,20,25]
    y_pred = choose_best_KNN(train_x,train_y,test_x,k_fold,neighbour_list)
    # Create dummy test output values
    predicted_y = y_pred
    # Output file location
    file_name = '../Predictions/PowerOutput/power_plant_KNN.csv'
    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    
    
    file2 =open('PPlogfile_KNN.txt','w')
    file2.write(str(error_mat)+'\n')
    file2.write(str(time_mat))
    file2.close()
    
   #######plot
    avg_time=[]
    for i in time_mat:
        avg_time.append(i[k_fold])
    graph=plt.plot(neighbour_list,avg_time,'rs')
    plt.ylabel('Avg Time for Validation(ms)')
    plt.xlabel('NUmber of neighbours')
    plt.title('Power dataset (Model=KNN)')
    plt.show()
    
    error_mat[:]=[]
    time_mat[:]=[]
    min_e=10000000
    min_n=3
    avg_time[:]=[]
    
    print('\n\n')
    

    
def indoor_localization_KNN():
    train_x, train_y, test_x = read_data_localization_indoors()
    print('Train=', train_x.shape)
    print('Test=', test_x.shape)
    k_fold=5
    neighbour_list=[3,5,10,20,25]
    y_pred = choose_best_KNN(train_x,train_y,test_x,k_fold,neighbour_list)
    # Create dummy test output values
    predicted_y = y_pred
    # Output file location
    file_name = '../Predictions/IndoorLocalization/indoor_localization_KNN.csv'
    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    
    file2 =open('ILlogfile_KNN.txt','w')
    file2.write(str(error_mat)+'\n')
    file2.write(str(time_mat))
    file2.close()
    
    #######plot
    avg_time=[]
    for i in time_mat:
        avg_time.append(i[k_fold])
    graph=plt.plot(neighbour_list,avg_time,'rs')
    plt.ylabel('Avg Time for Validation(ms)')
    plt.xlabel('NUmber of neighbours')
    plt.title('Indoor Localization dataset (Model=KNN)')
    plt.show()
    
    error_mat[:]=[]
    time_mat[:]=[]
    min_e=10000000
    min_n=3
    avg_time[:]=[]
    
    print('\n\n')

def indoor_localization_best():
    train_x, train_y, test_x = read_data_localization_indoors()
    print('Train=', train_x.shape)
    print('Test=', test_x.shape)
    print ('Best model for Indoor Localization Dataset')
    parameter=9
    print ("Training with full train data with Model: KNN for n="+str(parameter))
    neigh = KNeighborsRegressor(n_neighbors=parameter)
    neigh = neigh.fit(train_x,train_y)
    y_hat = neigh.predict(train_x)
    e = compute_error(y_hat,train_y)
    print ("MAE train error="+str(e))
    y_pred = neigh.predict(test_x)
    # Create dummy test output values
    predicted_y = y_pred
    # Output file location
    file_name = '../Predictions/IndoorLocalization/best.csv'
    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    
    error_mat[:]=[]
    time_mat[:]=[]
    min_e=10000000
    min_n=3
    
    print('\n\n')
    
    
    
#############################################################################################

############################################################################ DECISION TREES






    
def DT(neighbors,train_x,train_y):
    neigh = KNeighborsRegressor(n_neighbors=neighbors)
    neigh = neigh.fit(train_x,train_y)
    return (neigh)

def crossval_kfold_DT(k,max_depth,train_x,train_y):
    global error_mat
    global depth_for_min_e
    global min_e
    time_taken=[]
    kf = KFold(n_splits=k,shuffle=True)
    kf = kf.split(train_x,train_y)
    
    e=[]
    for train in kf:
        training_x=[]
        training_y=[]
        testing_x =[]
        testing_y=[]
        start_time=time.time()
        train_indices = train[0]
        test_indices = train[1]
        for idx in train_indices:
            training_x.append(train_x[idx])
            training_y.append(train_y[idx])
        for idx in test_indices:
            testing_x.append(train_x[idx])
            testing_y.append(train_y[idx]) 
            
        clf = tree.DecisionTreeRegressor(criterion='mae',max_depth=max_depth)
        clf = clf.fit(training_x,training_y)
        y_hat = clf.predict(testing_x)
        e.append(compute_error(y_hat,testing_y))
        end_time=time.time()
        delta_time = (end_time-start_time)*1000
        time_taken.append(delta_time)
        
        del (clf)
        del (training_x,training_y,testing_x,testing_y)
    
    avg_time = sum(time_taken)/float(len(time_taken))
    time_taken.append(avg_time)
    time_mat.append(time_taken)
    
    
    avg_e = sum(e)/float(len(e))
    e.append(avg_e)
    error_mat.append(e)
    print (str(e))
    
    min_e = min(min_e,avg_e)
    if(min_e==avg_e): depth_for_min_e = max_depth



def choose_best_DT(train_x,train_y,test_x,depths,k_fold):
    global error_mat
    global depth_for_min_e
    global min_e
    print (str(k_fold)+" fold cross validation:")
    for max_depth in depths:
        print ("Decision Tree regression training for depth="+str(max_depth)+":")
        crossval_kfold_DT(k_fold,max_depth,train_x,train_y)
    #print (error_mat)
    avg_errors=[]
    for i in error_mat:
        avg_errors.append(i[k_fold])
    print ('Average erros:'+str(avg_errors))
    # retraining with full training data
    print ("retraining with full train data for n="+str(depth_for_min_e))
    clf = tree.DecisionTreeRegressor(criterion='mae',max_depth=depth_for_min_e)
    clf = clf.fit(train_x,train_y)
    y_hat = clf.predict(train_x)
    e = compute_error(y_hat,train_y)
    print ("MAE train error="+str(e))
    y_pred = clf.predict(test_x)
    return y_pred

def power_plant_DT(k_fold):
    train_x, train_y, test_x = read_data_power_plant()
    print('Train=', train_x.shape)
    print('Test=', test_x.shape)   
    depths = [3,6,9,12,15]
    y_pred = choose_best_DT(train_x,train_y,test_x,depths,k_fold)
    
    #######plot
    avg_time=[]
    for i in time_mat:
        avg_time.append(i[k_fold])
    graph=plt.plot(depths,avg_time,'rs')
    plt.ylabel('Avg Time for Validation(ms)')
    plt.xlabel('Depth of Tree')
    plt.title('Power plant dataset (Model=Decision Tree)')
    plt.show()
    
    ######writing predictions to CSV file
    predicted_y = y_pred
    file_name = '../Predictions/PowerOutput/power_plant_DT.csv'
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    
    #######logs
    file2 =open('PPlogfile_DT.txt','w')
    file2.write(str(error_mat)+'\n\n')
    file2.write(str(time_mat)+'\n\n')
    file2.write(str(avg_time)+'\n')
    file2.close()
    
    error_mat[:]=[]
    time_mat[:]=[]
    avg_time[:]=[]
    min_e=10000000
    depth_for_min_e=3
    
    print('\n\n')


def indoor_localization_DT(k_fold):
    train_x, train_y, test_x = read_data_localization_indoors()
    print('Train=', train_x.shape)
    print('Test=', test_x.shape)
    depths = [3,6,9,12,15]
    y_pred = choose_best_DT(train_x,train_y,test_x,depths,k_fold)
    
    #####plot avg time
    avg_time=[]
    for i in time_mat:
        avg_time.append(i[k_fold])
    plt.plot(depths,avg_time,'bo')
    plt.ylabel('Avg Time for Validation')
    plt.xlabel('Depth of Tree')
    plt.title('Indoor Localisation Dataset')
    plt.show()
    
    #####write predictions in CSV file
    predicted_y = y_pred
    file_name = '../Predictions/IndoorLocalization/indoor_localization_DT.csv'
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    
    #####log file
    file2 =open('ILlogfile_DT.txt','w')
    file2.write(str(error_mat)+'\n\n')
    file2.write(str(time_mat)+'\n\n')
    file2.write(str(avg_time)+'\n')
    file2.close()
    
    error_mat[:]=[]
    time_mat[:]=[]
    avg_time[:]=[]
    min_e=10000000
    depth_for_min_e=3
    
    print('\n\n')
     

def power_plant_best():
    train_x, train_y, test_x = read_data_power_plant()
    print('Train=', train_x.shape)
    print('Test=', test_x.shape)
    print ('Best model for Power Output Dataset')
    parameter = 13
    print ("Training with full train data: Model=Decision Tree for depth="+str(parameter))
    clf = tree.DecisionTreeRegressor(criterion='mse',max_depth=parameter)
    clf = clf.fit(train_x,train_y)
    y_hat = clf.predict(train_x)
    e = compute_error(y_hat,train_y)
    print ("MAE train error="+str(e))
    y_pred = clf.predict(test_x)
    # Create dummy test output values
    predicted_y = y_pred
    # Output file location
    file_name = '../Predictions/PowerOutput/best.csv'
    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    
    error_mat[:]=[]
    time_mat[:]=[]
    min_e=10000000
    depth_for_min_e=3
    
    print('\n\n')
    
##########################################################################################################
##########################################################################################################
print('\n\n')
##MAIN

power_plant_DT(5)                     #k-fold=5
power_plant_KNN()
power_plant_LinearModel(5)            #k-fold=5

indoor_localization_DT(5)             #k-fold=5
indoor_localization_KNN()
indoor_localization_LinearModel(5)    #k-fold=5

power_plant_best()
indoor_localization_best()



