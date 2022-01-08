import random as random
import pickle
import os

# 경로 재설정 필요 : "/현재 Code File이 있는 Full Path/Code"
os.chdir("/../Code")
from matplotlib import pyplot as plt
import e3_OCF_B_S1 as OCF_B
from google.colab import files
import tensorflow as tf
import cupy as cp

def print_graph(result, term, dataName, metric):
    x = list(range(0, term+1))
    fig = plt.figure()
    g = fig.subplots()
    for model in result.keys():
        y = list(result[model][metric].values())
        g.plot(x, y, linestyle=":", marker="o", color=result[model]["RGB"], label=model)

    g.set_xlabel("Terms")
    g.set_ylabel(metric)
    g.legend()
    plt.xticks(x)
    plt.title(dataName)
    codeDirectory = os.path.dirname(os.getcwd())
    filePath = os.path.join(codeDirectory, "GraduationCode", "ResultGraph", "["+metric+"] "+dataName[0:-4]+".png")
    plt.savefig(filePath, dpi=100)

'''
userLimit : userID 몇번 부터 몇번까지 sampling 할지 (list형)
itemLimit : itemID 몇번 부터 몇번까지 sampling 할지 (list형)
'''
def experiment_by_model(models, epochs, NlatentUsers, NlatentItems, NlatentMFs, neg_cases, term, fileName, userLimit, itemLimit, graph_list):
    data = dict()

    for model in models:
        for epoch in epochs:
            for Nlu in NlatentUsers:
                for Nli in NlatentItems:
                    for Nmf in NlatentMFs:
                        for n_case in neg_cases:
                            if model is 'GMF':
                                name = "GMF(e:" + str(epoch) + "/lmf:" + str(Nmf) + "/n:" + str(n_case) + ")"
                                params = {
                                "model" : 'GMF',
                                "epoch" : epoch,
                                "NlatentMF" : Nmf,
                                "test_size" : 0.3,
                                "neg_case" : n_case,
                                "term" : term
                            }
                            elif model is 'MLP':
                                name = "MLP(e:" + str(epoch) + "/lu:" + str(Nlu) + "/li:" + str(Nli) + "/n:" + str(n_case) + ")"
                                params = {
                                "model" : 'MLP',
                                "epoch" : epoch,
                                "NlatentUser" : Nlu,
                                "NlatentItem" : Nli,
                                "test_size" : 0.3,
                                "neg_case" : n_case,
                                "term" : term
                            }
                            elif model is 'NeuMF':
                                name = "NeuMF(e:" + str(epoch) + "/lu:" + str(Nlu) + "/li:" + str(Nli) + "/lmf:" + str(Nmf) + "/n:" + str(n_case) + ")"
                                params = {
                                "model" : 'NeuMF',
                                "epoch" : epoch,
                                "NlatentUser" : Nlu,
                                "NlatentItem" : Nli,
                                "NlatentMF" : Nmf,
                                "test_size" : 0.3,
                                "neg_case" : n_case,
                                "term" : term
                            }
                            else:
                                raise ModelError('unexpected model name')

                            print("\n")
                            print("Start Running : ", name)
                            data[name] = OCF_B.run(fileName, userLimit, itemLimit, params)

                            # save result data
                            codeDirectory = os.path.dirname(os.getcwd())
                            filePath = os.path.join(codeDirectory, "GraduationCode", "result_data.pickle")
                            with open(filePath,'wb') as fw:
                                pickle.dump(data, fw)

    print("[ Start printing Graph ... ]")
    print(data)
    '''
    data = dict{
        model_name : dict{
            "RGB"  : [float, float, float]
            "AUC"  : dict{0:float, 1:float, 2:float, ..., term:float}
            "RMSE" : dict{0:float, 1:float, 2:float, ..., term:float}
            ....
        }
        ...
    }
    '''
    for metric in graph_list:
        print(" printing : ", metric)
        print_graph(data, term, fileName, metric)
    print("[ Printing Graph Finished! ]")


if __name__ == "__main__":

    # Epoch를 짧게 잡고 term를 길게하는게 좋은지?
    # Epoch를 길게 잡고 term를 짧게하는게 좋은지?

    '''
    [ Hyperparameter Tuning ]
    * 주의 : empty list가 있으면 안됨
    * 주의 : 경로 재설정 필요 (line 6)
    '''
    models = ['GMF']   # {'GMF' | 'MLP' | 'NeuMF'}
    epochs = [10]
    NlatentUsers = [3] # 해보고 싶은 latent user들 --> {'MLP' | 'NeuMF'}의 경우
    NlatentItems = [3] # 해보고 싶은 latent item들 --> {'MLP' | 'NeuMF'}의 경우
    NlatentMFs = [4]   # 해보고 싶은 latent mf들   --> {'GMF' | 'NeuMF'}의 경우
    terms = 10

    # 프린트하고 싶은 결과지표들
    graph_list = ["AUC", "RMSE", "Positive_RMSE", "Negative_RMSE", "Positive_Precision", "Negative_Precision", "Positive_Recall", "Negative_Recall"]

    neg_case_epi = [250000, 500000, 750000]
    experiment_by_model(models, epochs, NlatentUsers, NlatentItems, NlatentMFs, neg_case_epi, terms, "Epinions_oneClass.csv", [1, 49289], [1, 139738], graph_list)

    # neg_case_beauty = [430, 860, 1290]
    # experiment_by_model(models, epochs, NlatentUsers, NlatentItems, NlatentMFs, neg_case_beauty, terms, "beauty_fixed.csv", [1, 50], [1, 59], graph_list)

    # neg_cases_job = [3000, 6000, 9000]
    # experiment_by_model(models, epochs, NlatentUsers, NlatentItems, NlatentMFs, neg_cases_job, terms, "JobMatching.csv", [1, 926], [1, 12441], graph_list)

    # neg_cases_mov = [25000, 50000, 75000]
    # experiment_by_model(models, epochs, NlatentUsers, NlatentItems, NlatentMFs, neg_cases_mov, terms, "MovieLens100K_oneClass.csv", [1, 943], [1, 1682], graph_list)

    # neg_cases_rest = [240, 480, 720]
    # experiment_by_model(models, epochs, NlatentUsers, NlatentItems, NlatentMFs, neg_cases_rest, terms, "restaurant.csv", [0, 137], [0, 129], graph_list)

    # neg_cases_inter = [2740, 5480, 8220]
    # experiment_by_model(models, epochs, NlatentUsers, NlatentItems, NlatentMFs, neg_cases_inter, terms, "interaction.csv", [0, 2986], [0, 1894], graph_list)
