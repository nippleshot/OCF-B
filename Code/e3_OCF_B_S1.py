from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from math import sqrt
import random as random
import TrainingModule as TRM
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import cupy as cp
cp.cuda.Device(0).use()

def genNegative(num, added_pair, added_np_train, userID_limit, itemID_limit):
    negcnt = 0
    while negcnt < num:
        user_random = cp.random.randint(userID_limit[0], userID_limit[1])
        item_random = cp.random.randint(itemID_limit[0], itemID_limit[1])

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

'''
 params : dict{
            model         : str {'GMF' | 'MLP' | 'NeuMF'}
            epoch         : int
            NlatentUser   : int {IF USING 'MLP' | 'NeuMF'}
            NlatentItem   : int {IF USING 'MLP' | 'NeuMF'}
            NlatentMF     : int {IF USING 'GMF' | 'NeuMF'}
            test_size     : float
            neg_case      : int
            term          : int
        }
'''
def run(fileName, userID_limit, itemID_limit, params) :
    numNegaCase = params["neg_case"]

    aucs_dict = dict()
    rmse_dict = dict()
    pos_rmse_dict = dict()
    nega_rmse_dict = dict()
    precision_1_dict = dict()
    precision_0_dict = dict()
    recall_1_dict = dict()
    recall_0_dict = dict()

    def to_dictSet(aucs_dict, rmse_dict, pos_rmse_dict, nega_rmse_dict, precision_1_dict, precision_0_dict, recall_1_dict, recall_0_dict):
        data = {
            "RGB" : [random.random(), random.random(), random.random()],
            "AUC" : aucs_dict,
            "RMSE": rmse_dict,
            "Positive_RMSE" : pos_rmse_dict,
            "Negative_RMSE" : nega_rmse_dict,
            "Positive_Precision" : precision_1_dict,
            "Negative_Precision" : precision_0_dict,
            "Positive_Recall" : recall_1_dict,
            "Negative_Recall" : recall_0_dict
        }
        return data

    def update_dict(icnt, aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0):
        aucs_dict.update({icnt : aucs})
        rmse_dict.update({icnt : rmse})
        pos_rmse_dict.update({icnt : pos_rmse})
        nega_rmse_dict.update({icnt : nega_rmse})
        precision_1_dict.update({icnt: precision_1})
        precision_0_dict.update({icnt: precision_0})
        recall_1_dict.update({icnt: recall_1})
        recall_0_dict.update({icnt: recall_0})

    def do_train(arg, newdataset):
        if arg["model"] is 'GMF':
            return TRM.trainMFModel(newdataset, arg["epoch"], arg["NlatentMF"], arg["test_size"])
        elif arg["model"] is 'MLP':
            return TRM.trainMLPModel(newdataset, arg["epoch"], arg["NlatentUser"], arg["NlatentItem"], arg["test_size"])
        elif arg["model"] is 'NeuMF':
            return TRM.trainNeuMFModel(newdataset, arg["epoch"], arg["NlatentUser"], arg["NlatentItem"], arg["NlatentMF"], arg["test_size"])
        else:
            raise ModelError('unexpected model name')

    codeDirectory = os.path.dirname(os.getcwd())
    dataPath = os.path.join(codeDirectory, "GraduationCode", "Data", fileName)

    posi_dataset = pd.read_csv(dataPath, names="user_id,item_id,rating,timestamp".split(","))
    nnp_train = posi_dataset.to_numpy()
    np_train = cp.asarray(nnp_train)
    posi_pair = np_train[0:, :2]  # .csv에서 각 row의 userID, movieID만 빼내서 형성된 배열
    print("No of positive cases = ", len(np_train))

    # posi_pair에 없는 부정사례를 numNegaCase 건 생성하여 np_train에 추가
    print("No of negative cases = ", numNegaCase)
    added_np_train = genNegative(numNegaCase, posi_pair, np_train, userID_limit, itemID_limit)
    print("No of total training cases = ", len(added_np_train))

    added_np_train = cp.asnumpy(added_np_train)
    newdataset = pd.DataFrame(added_np_train, columns=['user_id', 'item_id', 'rating', 'timestamp'])

    # 1차 훈련:
    _, model, test, train = do_train(params, newdataset)
    aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0 = evaluate(model, test)
    icnt = 0
    update_dict(icnt, aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0)

    icnt = 1
    while icnt <= params["term"]:
        print("*************************** TERM : ", icnt, " ***************************")
        # train과 test를 합치고나서 긍정사례(posiSet)와 부정사례(negaSet) 셋으로 구분
        wdataset = pd.concat([train, test])
        groups = wdataset.groupby(wdataset.rating)
        posiSet = groups.get_group(1)
        negaSet = groups.get_group(0)

        # 부정사례셋에 대한 예측오차 계산
        predictScore = model.predict([negaSet.user_id, negaSet.item_id])
        negaSet['error'] = predictScore

        # 예측오차가 0.5이상인 부정사례 n개를 제거
        negaSet.drop(negaSet[negaSet.error > 0.5].index, inplace=True)

        print("예측오차가 큰 부정사례 ", (numNegaCase - len(negaSet)), " 가 제거되고, 부정사례 ", len(negaSet), " 건이 훈련데이터에 남음! ")
        negaSet.drop('error', axis=1,  inplace=True)

        # 부족한 부정사례를 다시 무작위 생성하여 최종 훈련데이터 구성 및 재훈련
        remain_train = pd.concat([posiSet, negaSet])
        # genNegative 들어갈때 파라메터 중에 데이터가 다시 numpy --> cupy로 변형
        remain_train = cp.asarray(remain_train)
        re_train = genNegative((numNegaCase - len(negaSet)), posi_pair,
                               remain_train, userID_limit, itemID_limit)  # posi_pair에 없는 부정사례를 좀전에 제거한 n건 만큼 생성하여 np_train에 추가
        print("훈련데이터 건수 : ", len(re_train))
        # genNegative에서 나온 데이터 다시 cupy --> numpy로 변형
        re_train = cp.asnumpy(re_train)
        re_dataset = pd.DataFrame(re_train, columns=['user_id', 'item_id', 'rating', 'timestamp'])
        _, model, test, train = do_train(params, re_dataset)

        # 성능평가 및 결과 출력
        aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0 = evaluate(model, test)
        update_dict(icnt, aucs, rmse, pos_rmse, nega_rmse, precision_1, precision_0, recall_1, recall_0)
        icnt = icnt + 1

    return to_dictSet(aucs_dict, rmse_dict, pos_rmse_dict, nega_rmse_dict, precision_1_dict, precision_0_dict, recall_1_dict, recall_0_dict)
