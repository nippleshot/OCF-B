from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from math import sqrt
import TrainingModule as TRM
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
os.chdir("/content/drive/MyDrive/GraduationLab/Graduation/GraduationCode")
from tqdm import tqdm
pd.set_option('display.max_columns', None)
from keras.models import load_model

codeDirectory = os.path.dirname(os.getcwd())

def makeInitialModel(fileName) :
    dataPath = os.path.join(codeDirectory, "GraduationCode", "Data", fileName)
    dataset = pd.read_csv(dataPath, names="user_id,item_id,rating,timestamp".split(","))
    TRM.zeroInjectionMF(dataset, 50, 3)

def save_ZeroPredicted():
  mfModelPath = os.path.join(codeDirectory, "GraduationCode", "MFModel", "best_model.h5")
  model = load_model(mfModelPath)

  zeroListPath = os.path.join(codeDirectory, "GraduationCode", "Data", "MovieLens100K_ZeroList.csv")
  zeroList_data = pd.read_csv(zeroListPath, names="user,item,rating,timestamp".split(","))

  print(zeroList_data.head(15))
  print("\n")
  predictScores = model.predict([zeroList_data.user, zeroList_data.item])
  predicted = pd.DataFrame(data=predictScores, columns=['predicted'])
  print(predicted.head(15))
  zeroList_data.drop(['rating','timestamp'], axis='columns', inplace=True)
  print(zeroList_data.head(15))
  zeroList_result = pd.concat([zeroList_data, predicted], axis=1)
  zeroList_result.drop(index=0, axis=0, inplace=True)
  zeroList_result.astype({'user':int, 'item':int})
  zeroList_result.sort_values(by=['predicted'], ascending=True, inplace=True)
 
  zeroResultPath = os.path.join(codeDirectory, "GraduationCode", "Data", "MovieLens100K_ZeroSorted.csv")
  zeroList_result.to_csv(zeroResultPath, header=False, index=False)

def get_NegativeCase(num):
  zeroPath = os.path.join(codeDirectory, "GraduationCode", "Data", "MovieLens100K_ZeroSorted.csv")
  zeroPredicted = pd.read_csv(zeroPath , names="user_id,item_id,predicted".split(","))
  zeroPredicted = zeroPredicted.iloc[ 0:num , : ]
  zeroPredicted['predicted'] = 0
  zeroPredicted.rename(columns={'predicted': 'rating'}, inplace=True)
  return zeroPredicted

def get_PositiveCase():
  onePath = os.path.join(codeDirectory, "GraduationCode", "Data", "MovieLens100K_oneClass.csv")
  oneData = pd.read_csv(onePath, names="user_id,item_id,rating,timestamp".split(","))
  oneData.drop('timestamp', axis=1, inplace=True)
  return oneData

def evaluate(model, test):
    predictScore = model.predict([test.user_id, test.item_id])

    y_true_rating = test.rating.values
    y_true = y_true_rating
    y_hat_rating = [x[0] for x in predictScore]
    y_hat = [1 if x[0] > 0.5 else 0 for x in predictScore]
    print(classification_report(y_true, y_hat))
    
    aucs = roc_auc_score(y_true, y_hat_rating)
    rmse = sqrt(mean_squared_error(y_true_rating, y_hat_rating))

    precision_1 = precision_score(y_true, y_hat, pos_label=1)
    recall_1 = recall_score(y_true, y_hat, pos_label=1)
    precision_0 = precision_score(y_true, y_hat, pos_label=0)
    recall_0 = recall_score(y_true, y_hat, pos_label=0)

    y_hat_rating = np.array(y_hat_rating)
    pi = np.where(y_true_rating[:] == 1)
    y_true_rating1 = y_true_rating[pi[0]]
    y_hat_rating1 = y_hat_rating[pi[0]]
    
    pi = np.where(y_true_rating[:] == 0)
    y_true_rating0 = y_true_rating[pi[0]]
    y_hat_rating0 = y_hat_rating[pi[0]]

    pos_rmse = sqrt(mean_squared_error(y_true_rating1, y_hat_rating1))
    nega_rmse = sqrt(mean_squared_error(y_true_rating0, y_hat_rating0))

    return aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0

def print_result(negNum, aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0):
    print("=================== RESULT (n=",negNum,") ===================")
    print("aucs               = ", aucs)
    print("rmse               = ", rmse)
    print("positive rmse  = ", pos_rmse)
    print("negative rmse = ", nega_rmse)
    print("precision_1     = ", precision_1)
    print("recall_1           = ", recall_1)
    print("precision_0     = ", precision_0)
    print("recall_0           = ", recall_0)
    print("\n")

def experiment2_by_negaCaseNum(negaCaseNum_list):
  positive_data = get_PositiveCase()
  for negNum in negaCaseNum_list:
    dataset = pd.concat([positive_data, get_NegativeCase(negNum)], axis=0)
    history, model, test, train = TRM.trainMFModel(dataset, 80, 3, 0.3)
    aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0 = evaluate(model, test)
    print_result(negNum, aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0)

if __name__ == "__main__":
  negaCaseNum_list = [20000, 50000, 80000]
  experiment2_by_negaCaseNum(negaCaseNum_list)







