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
import cupy as cp
cp.cuda.Device(0).use()

"""
< genNegative >
parameters
1. num : Number of "0" cases to generate
2. added_pair : User-Movie rating case가 "1"인 [userID, movieID]로 형성된 배열 -- a.k.a 생성되면 안되는 a(ID난수 pair)의 list
3. added_np_train : 가상데이터가 추가되야되는 훈련데이터

return 
1. added_np_train : num개의 "0" cases 가상데이터가 추가된 훈련데이터

variables
a : [userID 1~943 난수 , movieID 1~1682 난수] -- a.k.a ID난수 pair
xu : added_pair의 각 userID 중에 a의 userID와 일치하고 added_pair의 각 movieID 중에 a의 movieID와 일치한 결과물
"""

def genNegative(num, added_pair, added_np_train, userID_limit, itemID_limit):
    negcnt = 0
    print("genNegative() : Start generating random negative cases")
    while negcnt < num: 
        # MovieLens100K
        user_random = cp.random.randint(1, userID_limit)
        item_random = cp.random.randint(1, itemID_limit)

        # Epinion
        # user_random = cp.random.randint(1, 49290)
        # item_random = cp.random.randint(1, 139738)

        xu = cp.where((added_pair[:,0] == user_random) & (added_pair[:,1] ==  item_random))
        # 일치하는 결과물이 없다면
        #   [userID 난수 , movieID 난수, 0, xxxx]를 added_np_train에다가 추가
        #   [userID 난수 , movieID 난수]를 added_pair에다가 추가
        if len(xu[0]) < 1 :
            new_added_pair = cp.array([user_random, item_random])
            new_added_np_train = cp.array([user_random,  item_random, cp.array(0), cp.array(111111111)])
            new_added_pair = cp.reshape(new_added_pair, (1,2))
            new_added_np_train = cp.reshape(new_added_np_train, (1,4))

            added_np_train = cp.concatenate((added_np_train, new_added_np_train), axis = 0)
            added_pair = cp.concatenate((added_pair, new_added_pair), axis = 0)
            negcnt = negcnt + 1
            if negcnt%10000==0:
              print(negcnt,"/",num) 
    return added_np_train


def evaluate(model, test):
    predictScore = model.predict([test.user_id, test.item_id])

    y_true_rating = test.rating.values
    y_true = y_true_rating #[1 if x > 3 else 0 for x in y_true_rating]
    y_hat_rating = [x[0] for x in predictScore]
    y_hat = [1 if x[0] > 0.5 else 0 for x in predictScore]
    
    print(classification_report(y_true, y_hat))
    
    aucs = roc_auc_score(y_true, y_hat_rating)
    rmse = sqrt(mean_squared_error(y_true_rating, y_hat_rating))
    print("aucs = ", aucs)
    print("rmse = ", rmse)

    precision_1 = precision_score(y_true, y_hat, pos_label=1)
    recall_1 = recall_score(y_true, y_hat, pos_label=1)
    print("precision_1 = ", precision_1)
    print("recall_1    = ", recall_1)

    precision_0 = precision_score(y_true, y_hat, pos_label=0)
    recall_0 = recall_score(y_true, y_hat, pos_label=0)
    print("precision_0 = ", precision_0)
    print("recall_0    = ", recall_0)

    y_hat_rating = np.array(y_hat_rating)
    pi = np.where(y_true_rating[:] == 1)
    y_true_rating1 = y_true_rating[pi[0]]
    y_hat_rating1 = y_hat_rating[pi[0]]
    
    pi = np.where(y_true_rating[:] == 0)
    y_true_rating0 = y_true_rating[pi[0]]
    y_hat_rating0 = y_hat_rating[pi[0]]

    pos_rmse = sqrt(mean_squared_error(y_true_rating1, y_hat_rating1))
    print("positive rmse = ", pos_rmse)
    nega_rmse = sqrt(mean_squared_error(y_true_rating0, y_hat_rating0))
    print("negative rmse = ", nega_rmse)

    return aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0


def run(numNegaCase, iter, fileName, userID_limit, itemID_limit) :
    aucs_dict = dict(n=numNegaCase)
    rmse_dict = dict(n=numNegaCase)
    pos_rmse_dict = dict(n=numNegaCase)
    nega_rmse_dict = dict(n=numNegaCase)
    precision_1_dict = dict(n=numNegaCase)
    precision_0_dict = dict(n=numNegaCase)
    recall_1_dict = dict(n=numNegaCase)
    recall_0_dict = dict(n=numNegaCase)

    # Number of negative cases to generate
    # numNegaCase = 80000
    # iter = 10
 
    codeDirectory = os.path.dirname(os.getcwd())
    dataPath = os.path.join(codeDirectory, "GraduationCode", "Data", fileName)

    posi_dataset = pd.read_csv(dataPath, names="user_id,item_id,rating,timestamp".split(","))
    nnp_train = posi_dataset.to_numpy()
    np_train = cp.asarray(nnp_train)
    posi_pair = np_train[0:, :2]  # .csv에서 각 row의 userID, movieID만 빼내서 형성된 배열
    print("No of positive cases = ", len(np_train))

    # posi_pair에 없는 부정사례를 numNegaCase 건 생성하여 np_train에 추가
    added_np_train = genNegative(numNegaCase, posi_pair, np_train, userID_limit, itemID_limit)
    print("No of total training cases = ", len(added_np_train))
    added_np_train = cp.asnumpy(added_np_train)
    newdataset = pd.DataFrame(added_np_train, columns=['user_id', 'item_id', 'rating', 'timestamp'])

    # 1차 훈련: trainMFModel(dataset, epo, NlatentFactor, testSize)
    # Epinions
    # history, model, test, train = TRM.trainMFModel(re_dataset, 11, 3, 0.3)
    history, model, test, train = TRM.trainMFModel(newdataset, 20, 3, 0.3)
    # history, model, test, train = TRM.trainMFModel(newdataset, 80, 3, 0.3)
    aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0 = evaluate(model, test)
    icnt = 0
    aucs_dict.update({icnt : aucs})
    rmse_dict.update({icnt : rmse})
    pos_rmse_dict.update({icnt : pos_rmse})
    nega_rmse_dict.update({icnt : nega_rmse})
    precision_1_dict.update({icnt: precision_1})
    precision_0_dict.update({icnt: precision_0})
    recall_1_dict.update({icnt: recall_1})
    recall_0_dict.update({icnt: recall_0})


    icnt = 1
    while icnt <= iter:
        print("*************************** ITERATION : ", icnt)
        # train과 test를 합치고나서 긍정사례(posiSet)와 부정사례(negaSet) 셋으로 구분
        wdataset = pd.concat([train, test])
        groups = wdataset.groupby(wdataset.rating)
        posiSet = groups.get_group(1)
        negaSet = groups.get_group(0)

        # 부정사례 데이터셋에 대한 예측오차 계산
        predictScore = model.predict([negaSet.user_id, negaSet.item_id])
        negaSet['error'] = predictScore

        # 예측오차가 0.5이상인 부정사례 n개를 제거
        indexNames = negaSet[abs(negaSet['error']) > 0.5].index
        negaSet.drop(indexNames, inplace=True)
        print("예측오차가 큰 부정사례 ", (numNegaCase - len(negaSet)), " 가 제거되고, 부정사례 ", len(negaSet), " 건이 훈련데이터에 남음! ")
        negaSet.drop('error', inplace=True, axis=1)

        # 부족한 부정사례를 다시 무작위 생성하여 최종 훈련데이터 구성 및 재훈련
        remain_train = pd.concat([posiSet, negaSet])
        # genNegative 들어갈때 파라메터 중에 데이터가 다시 numpy --> cupy로 변형
        remain_train = cp.asarray(remain_train)
        re_train = genNegative((numNegaCase - len(negaSet)), posi_pair,
                               remain_train, userID_limit, itemID_limit)
        print("훈련데이터 건수 : ", len(re_train))
        # genNegative에서 나온 데이터 다시 cupy --> numpy로 변형
        re_train = cp.asnumpy(re_train)
        re_dataset = pd.DataFrame(re_train, columns=['user_id', 'item_id', 'rating', 'timestamp'])
        # history, model, test, train = TRM.trainMFModel(re_dataset, 80, 3, 0.3)
        history, model, test, train = TRM.trainMFModel(re_dataset, 20, 3, 0.3)

        # Epinions
        #history, model, test, train = TRM.trainMFModel(re_dataset, 11, 3, 0.3)

        # 성능평가 및 결과 출력
        aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0 = evaluate(model, test)
        aucs_dict.update({icnt : aucs})
        rmse_dict.update({icnt : rmse})
        pos_rmse_dict.update({icnt : pos_rmse})
        nega_rmse_dict.update({icnt : nega_rmse})
        precision_1_dict.update({icnt: precision_1})
        precision_0_dict.update({icnt: precision_0})
        recall_1_dict.update({icnt: recall_1})
        recall_0_dict.update({icnt: recall_0})
        icnt = icnt + 1

    return aucs_dict, rmse_dict, pos_rmse_dict, nega_rmse_dict, precision_1_dict, precision_0_dict, recall_1_dict, recall_0_dict




