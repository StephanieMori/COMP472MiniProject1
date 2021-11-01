#Task 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from pandas.api.types import CategoricalDtype #added so we can order the BP and Cholesterol
from sklearn.naive_bayes import GaussianNB
import io
from tabulate import tabulate
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits #from perceptron example
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score 


ds = pd.read_csv('/content/sample_data/test36.csv', index_col=0) # part 2
print('~~~~~~~~~~~~~~~~~ Drug.csv ~~~~~~~~~~~~~~~~~  ')
print(ds)
print('\n Drug document information: \n')
ds.info()

print('\n~~~~~~~~~~~~~~~~~ Plot distribution ~~~~~~~~~~~~~~~~~\n') # part 3
ds_category= ds['Drug'].value_counts(sort=True)
print(ds_category,'\n') #info about drug category
plt.plot(ds_category)
plt.savefig('/content/sample_data/drug-distribution.pdf')
plt.show()






print('\n~~~~~~~~~~~~~~~~~ Categorical/Nominal to numerical  ~~~~~~~~~~~~~~~~~\n') # part 4
#nominal features
ds=pd.get_dummies(ds, columns=['Sex'], drop_first=True) #optimize the number of col for gender
#ds=pd.get_dummies(ds, columns=['Drug'], drop_first=True)# is it nominal??????

# Categorical features
cleanup_nums={"BP":     {"HIGH": 1, "NORMAL": 2, "LOW": 3},
                "Cholesterol": {"HIGH": 1, "NORMAL": 2 },
                 "Drug": {"drugY": 1, "drugX": 2, "drugA":3, "drugB": 4, "drugC": 5}} #Drug is categorical if you consider the number of occurences in the plot distribution

CategoricalDtype(categories=["BP", "Cholesterol","Drug"], ordered=True)
ds = ds.replace(cleanup_nums)
print(ds)

print('\n~~~~~~~~~~~~~~~~~ Train and Test  ~~~~~~~~~~~~~~~~~\n') #part 5
Y= ds.Drug #target
X = ds.drop('Drug', axis=1)#data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size = None, test_size=None, random_state=None) # set default
print("\n X training set:\n")
print(X_train.head())
print(X_train.shape)

print("\n X testing set:\n")
print(X_test.head())
print(X_test.shape)

#open file for info (part8?)
openfile = open("/content/sample_data/drug-performance.txt", 'w')

class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
feature_columns = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']


print('\n~~~~~~~~~~~~~~~~~ Gaussian NB  ~~~~~~~~~~~~~~~~~\n')

nb = GaussianNB()
# predict probabilities for test set
nb_prob=nb.fit(X_train,Y_train)  
# predict crisp class for test set's imbalanced classifiers 
nb_prediction= nb.predict(X_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, nb_prediction)
print('\n Gaussian NB`s Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, nb_prediction, average='weighted', labels=np.unique(nb_prediction)) # added lables to prevent an UndefinedMetricWarning
print('\n Gaussian NB`s Precision: %f' % precision)
# recall: tp / (tp + fn)
openfile.write("\n ~~~~~~~~~~~~~~ Gaussian NB`s   ~~~~~~~~~~~~~\n")
recall = recall_score(Y_test, nb_prediction, average='weighted')
print('\n Gaussian NB`s Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f_11 = f1_score(Y_test, nb_prediction, average='macro')
print('\n Gaussian NB`s F1 score macro: %f' % f_11)
# f1: 2 tp / (2 tp + fp + fn)
f_12 = f1_score(Y_test, nb_prediction, average='weighted')
print('\n Gaussian NB`s F1 score weighted: %f' % f_12)

openfile.write("\n ~~~~~~~~~~~~~~ Confusion Matrix ~~~~~~~~~~~~~\n")
con_mat = confusion_matrix(Y_test, nb_prediction)
print('\n Gaussian NB`s Confusion Matrix: ')
print (con_mat)

openfile.write(tabulate(con_mat, class_columns, tablefmt="pipe", stralign='center'))
openfile.write('\n')

rows = ["accuracy", "macro-average F1", "weighted-average F1"]
accuracy = str(accuracy)
macro_f1 = str(f_11)
weighted_f1 = str(f_12)
showdata = pd.DataFrame([accuracy, macro_f1, weighted_f1], rows)
openfile.write('\n')
openfile.write(tabulate(showdata, tablefmt="pipe"))
openfile.write("\n")


print('\n~~~~~~~~~~~~~~~~~ Base DT ~~~~~~~~~~~~~~~~~\n')
dt = DecisionTreeClassifier()
# predict probabilities for test set
dt_prob=dt.fit(X_train,Y_train)  
# predict crisp class for test set's imbalanced classifiers 
dt_prediction= dt.predict(X_test)

openfile.write("\n ~~~~~~~~~~~~~~ Base DT ~~~~~~~~~~~~~\n")

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, dt_prediction)
print('\n Base DT`s Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, dt_prediction, average='weighted', labels=np.unique(dt_prediction)) # added lables to prevent an UndefinedMetricWarning
print('\n Base DT`s Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, dt_prediction, average='weighted')
print('\n Base DT`s Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f_11 = f1_score(Y_test, dt_prediction, average='macro')
print('\n Base DT`s F1 score macro: %f' % f_11)
# f1: 2 tp / (2 tp + fp + fn)
f_12 = f1_score(Y_test, dt_prediction, average='weighted')
print('\n Base DT`s F1 score weighted: %f' % f_12)


openfile.write("\n ~~~~~~~~~~~~~~ Confusion Matrix ~~~~~~~~~~~~~\n")
con_mat = confusion_matrix(Y_test, dt_prediction)
print('\n Base DT`s Confusion Matrix: ')
print (con_mat)

openfile.write(tabulate(con_mat, class_columns, tablefmt="pipe", stralign='center'))
openfile.write('\n')

rows = ["accuracy", "macro-average F1", "weighted-average F1"]
accuracy = str(accuracy)
macro_f1 = str(f_11)
weighted_f1 = str(f_12)
showdata = pd.DataFrame([accuracy, macro_f1, weighted_f1], rows)
openfile.write('\n')
openfile.write(tabulate(showdata, tablefmt="pipe"))
openfile.write("\n")


print('\n~~~~~~~~~~~~~~~~~ Top DT  ~~~~~~~~~~~~~~~~~\n')

openfile.write("\n ~~~~~~~~~~~~~~ Top DT ~~~~~~~~~~~~~\n")

parameters = {'criterion': ('entropy', 'gini'),
                'max_depth': (5, 10), 'min_samples_split': (2, 4, 6)}
topdt = GridSearchCV(DecisionTreeClassifier(), parameters)
topdt.fit(X_train, Y_train)
# predict with the best found params
topdt_predict = topdt.predict(X_test)

#param1={'criterion': ['gini','entropy'],
#       'max_depth':[3,4],
#       'min_sample_split':[1,2,3]}

#top_dt= GridSearchCV(DecisionTreeClassifier(), param1,cv=5)
#top_dt.fit(X_train, Y_train)
#top_dt.best_estimator_

# predict probabilities for test set
#top_dt_prob=top_dt.fit(X_train,Y_train) 
# predict crisp class for test set's imbalanced classifiers 
#top_dt_prediction= top_dt.predict(X_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, topdt_predict)
print('\n Top Decision Tree Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, topdt_predict, average='weighted', labels=np.unique(topdt_predict)) # added lables to prevent an UndefinedMetricWarning
print('\n Top Decision Tree Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, topdt_predict, average='weighted')
print('\n Top Decision Tree Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f_11 = f1_score(Y_test, topdt_predict, average='macro')
print('\n Top Decision Tree F1 score macro: %f' % f_11)
# f1: 2 tp / (2 tp + fp + fn)
f_12 = f1_score(Y_test, topdt_predict, average='weighted')
print('\n Top Decision Tree F1 score weighted: %f' % f_12)

openfile.write("\n ~~~~~~~~~~~~~~ Confusion Matrix ~~~~~~~~~~~~~\n")
con_mat = confusion_matrix(Y_test, topdt_predict)
print('\n Top Decision Tree Confusion Matrix: ')
print (con_mat)

openfile.write(tabulate(con_mat, class_columns, tablefmt="pipe", stralign='center'))
openfile.write('\n')

rows = ["accuracy", "macro-average F1", "weighted-average F1"]
accuracy = str(accuracy)
macro_f1 = str(f_11)
weighted_f1 = str(f_12)
showdata = pd.DataFrame([accuracy, macro_f1, weighted_f1], rows)
openfile.write('\n')
openfile.write(tabulate(showdata, tablefmt="pipe"))
openfile.write("\n")


# Gridsearch will find the best combination of hyper-parameters
#param_grid=dict(criterion= ['gini','entropy'],
#            max_depth=range(1,2),
#            min_sample_split=range(1,3))
#grid= GridSearchCV(top_dt,
#                   param_grid,
#                   cv=5,
#                   )

#grid.fit(X_train,Y_train)

#best_param=grid.best_params_
#best_estimate= grid.best_estimator_
#print('best parameters for the Top DT are: ', best_param)
#print('best Values for parameters for the Top DT are: ', best_estimate)


# creating more models
#accuracy = []
#macro_f1 = []
#weighted_f1 = []
#for i in range(10):

    # creating new models
    #top_DT_param = {'criterion': ('entropy', 'gini'),
    #                'max_depth': (2, 7), 'min_samples_split': (3, 6, 8)}

    #clf = [GaussianNB(),
      #     DecisionTreeClassifier(),
     #      GridSearchCV(DecisionTreeClassifier(), top_DT_param),
     #      ]
  #  precision_ = []
   # accuracy_ = []
  #  macro_ = []
  #  weighted_ = []

    # training and testing new models
    #for j in range(3):
        # training
    #    clf[j].fit(X, Y)
        # testing
     #   p_result.append(clf[j].predict(X_test))

    # recording scores
    #for i in range(3):
        
     #   accuracy_.append(accuracy_score(Y_test, precision_[i]))
     #   accuracy.append(accuracy_)
        
      #  macro_.append(f1_score(Y_test, precision_[i], average='macro'))
       # macro_f1.append(macro_)
        
       # weighted_temp.append(f1_score(Y_test, precision_[i], average='weighted'))
       # weighted_f1.append(weighted_)

# calculate the average and stddev
#accuracy = np.array(accuracy)
#macro_f1 = np.array(macro_f1)
#weighted_f1 = np.array(weighted_f1)

#mean_accuracy = np.mean(accuracy, axis=0)
#mean_macro_f1 = np.mean(macro_f1, axis=0)
#mean_weighted_f1 = np.mean(weighted_f1, axis=0)

#std_accuracy = np.std(accuracy, axis=0)
#std_macro_f1 = np.std(macro_f1, axis=0)
#std_weighted_f1 = np.std(weighted_f1, axis=0)

#columns = ["nb", "dt", "top_dt"]
#rows = ["mean_accuracy", "mean_macro_f1", "mean_weighted_f1", "std_accuracy", "std_macro_f1", "std_weighted_f1"]

#average_total = pd.DataFrame([mean_accuracy,
 #                    mean_macro_f1,
 #                    mean_weighted_f1,
 #                    std_accuracy,
  #                   std_macro_f1,
   #                  std_weighted_f1],
  #                   rows)
#openfile.write('\n')
#openfile.write('\n [       Ten times average       ] \n')
#openfile.write('\n')
#openfile.write(tabulate(t10average, columns, tablefmt="pipe"))

#openfile.close()


print('\n~~~~~~~~~~~~~~~~~ Perceptron  ~~~~~~~~~~~~~~~~~\n') 
openfile.write("\n ~~~~~~~~~~~~~~ Perceptron ~~~~~~~~~~~~~\n")

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)

# predict probabilities for test set
per_prob = perceptron.predict(X_test)
# predict crisp class for test set's imbalanced classifiers 
per_prediction = perceptron.predict(X_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, per_prediction)
print('\nPerceptron`s Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, per_prediction, average='weighted', labels=np.unique(per_prediction)) # added lables to prevent an UndefinedMetricWarning
print('\nPerceptron`s Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, per_prediction, average='weighted')
print('\nPerceptron`s Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f_11 = f1_score(Y_test, per_prediction, average='macro')
print('\nPerceptron`s F1 score macro: %f' % f_11)
# f1: 2 tp / (2 tp + fp + fn)
f_12 = f1_score(Y_test, per_prediction, average='weighted')
print('\nPerceptron`s F1 score weighted: %f' % f_12)

openfile.write("\n ~~~~~~~~~~~~~~ Confusion Matrix ~~~~~~~~~~~~~\n")
con_mat = confusion_matrix(Y_test, per_prediction)
print('\nPerceptron`s Confusion Matrix: ')
print (con_mat)

openfile.write(tabulate(con_mat, class_columns, tablefmt="pipe", stralign='center'))
openfile.write('\n')

rows = ["accuracy", "macro-average F1", "weighted-average F1"]
accuracy = str(accuracy)
macro_f1 = str(f_11)
weighted_f1 = str(f_12)
showdata = pd.DataFrame([accuracy, macro_f1, weighted_f1], rows)
openfile.write('\n')
openfile.write(tabulate(showdata, tablefmt="pipe"))
openfile.write("\n")



print('\n~~~~~~~~~~~~~~~~~ Base MLP  ~~~~~~~~~~~~~~~~~\n') 
openfile.write("\n ~~~~~~~~~~~~~~ Base MLB ~~~~~~~~~~~~~\n")

mlb=MLPClassifier(hidden_layer_sizes=(100,),max_iter=500, solver='sgd', activation='logistic')
mlb.fit(X_train, Y_train)

# predict probabilities for test set
mlb_prob = mlb.predict(X_test)
# predict crisp class for test set's imbalanced classifiers
mlb_prediction = mlb.predict(X_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, mlb_prediction)
print('\n Base MLB`s Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, mlb_prediction, average='weighted',labels=np.unique(mlb_prediction)) # added lables to prevent an UndefinedMetricWarning)
print('\n Base MLB`s Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, mlb_prediction, average='weighted')
print('\n Base MLB`s Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f_11 = f1_score(Y_test, mlb_prediction, average='macro')
print('\n Base MLB`s F1 score macro: %f' % f_11)
# f1: 2 tp / (2 tp + fp + fn)
f_12 = f1_score(Y_test, mlb_prediction, average='weighted')
print('\n Base MLB`s F1 score weighted: %f' % f_12)

openfile.write("\n ~~~~~~~~~~~~~~ Confusion Matrix ~~~~~~~~~~~~~\n")
con_mat = confusion_matrix(Y_test, mlb_prediction)
print('\n Base MLB`s Confusion Matrix: ')
print (con_mat)

openfile.write(tabulate(con_mat, class_columns, tablefmt="pipe", stralign='center'))
openfile.write('\n')

rows = ["accuracy", "macro-average F1", "weighted-average F1"]
accuracy = str(accuracy)
macro_f1 = str(f_11)
weighted_f1 = str(f_12)
showdata = pd.DataFrame([accuracy, macro_f1, weighted_f1], rows)
openfile.write('\n')
openfile.write(tabulate(showdata, tablefmt="pipe"))
openfile.write("\n")


print('\n~~~~~~~~~~~~~~~~~ TOP MLP  ~~~~~~~~~~~~~~~~~\n') 
openfile.write("\n ~~~~~~~~~~~~~~ Top MLB ~~~~~~~~~~~~~\n")

param2={'hidden_layer_sizes': [(30,50),(10,10,10)],
        'activation': ['identity','logistic','tanh','relu'],
        'solver': ['sgd','adam']}

top_mlb= GridSearchCV(MLPClassifier(),param2,n_jobs=10,cv=3)
top_mlb.fit(X_train, Y_train)
top_mlb.best_estimator_


# predict probabilities for test set
top_mlb_prob = top_mlb.predict(X_test)
# predict crisp class for test set's imbalanced classifiers
top_mlb_prediction = top_mlb.predict(X_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, top_mlb_prediction)
print('\n Top MLB`s Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, mlb_prediction, average='weighted',labels=np.unique(top_mlb_prediction)) # added lables to prevent an UndefinedMetricWarning)
print('\n Top MLB`s Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, top_mlb_prediction, average='weighted')
print('\n Top MLB`s Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f_11 = f1_score(Y_test, top_mlb_prediction, average='macro')
print('\n Top MLB`s F1 score macro: %f' % f_11)
# f1: 2 tp / (2 tp + fp + fn)
f_12 = f1_score(Y_test, top_mlb_prediction, average='weighted')
print('\n Top MLB`s F1 score weighted: %f' % f_12)

openfile.write("\n ~~~~~~~~~~~~~~ Confusion Matrix ~~~~~~~~~~~~~\n")
con_mat = confusion_matrix(Y_test, top_mlb_prediction)
print('\n Top MLB`s Confusion Matrix: ')
print (con_mat)

openfile.write(tabulate(con_mat, class_columns, tablefmt="pipe", stralign='center'))
openfile.write('\n')

rows = ["accuracy", "macro-average F1", "weighted-average F1"]
accuracy = str(accuracy)
macro_f1 = str(f_11)
weighted_f1 = str(f_12)
showdata = pd.DataFrame([accuracy, macro_f1, weighted_f1], rows)
openfile.write('\n')
openfile.write(tabulate(showdata, tablefmt="pipe"))
openfile.write("\n")
 
openfile.close()


