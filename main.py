import random as random
import os
os.chdir("/content/drive/MyDrive/GraduationLab/Graduation/GraduationCode")
from matplotlib import pyplot as plt
import OCF_B_V1 as OCF_B
from google.colab import files
import tensorflow as tf
import cupy as cp

def printGraph(eval_index, iter, dataName, eval_datas):
    iteration_x = list(range(0, iter+1))
    fig = plt.figure()
    eval_graph = fig.subplots()
    for data in eval_datas:
        print("\nDoing printGraph() for문 data : ", eval_index, " --> ", data)
        neg_Num = "n="+str(data["n"])
        del(data["n"])
        iteration_y = list(data.values())
        if neg_Num == "n=20000" or neg_Num == "n=250000" or neg_Num == "n=730000":
            rgb = [0.7, 0.7, 0.2]
        elif neg_Num == "n=50000" or neg_Num == "n=500000" or neg_Num == "n=1400000":
            rgb = [0.7, 0.4, 1]
        elif neg_Num == "n=80000" or neg_Num == "n=750000" or neg_Num == "n=2130000":
            rgb = [0.3, 1, 0.4]
        else:
            rgb = [random.random(), random.random(), random.random()]

        eval_graph.plot(iteration_x, iteration_y, color=rgb, label=neg_Num)

    eval_graph.set_xlabel("Iteration")
    eval_graph.set_ylabel(eval_index)
    plt.xticks(iteration_x)
    plt.legend()
    plt.title(dataName)
    codeDirectory = os.path.dirname(os.getcwd())
    filePath = os.path.join(codeDirectory, "GraduationCode", "ResultGraph", "["+eval_index+"] "+dataName[0:-4] + "_iter" + str(iter)+".png")
    plt.savefig(filePath, dpi=100)

def experiment1_by_negaCaseNum(negaCaseNum_list, iterNum, fileName, userLimit, itemLimit):
    aucs_pkg = []
    rmse_pkg = []
    pos_rmse_pkg = []
    nega_rmse_pkg = []
    precision_1_pkg = []
    precision_0_pkg = []
    recall_1_pkg = []
    recall_0_pkg = []


    for negaCaseNum in negaCaseNum_list:
        aucs_dict, rmse_dict, pos_rmse_dict, nega_rmse_dict, precision_1_dict, precision_0_dict, recall_1_dict, recall_0_dict = OCF_B.run(negaCaseNum, iterNum, fileName, userLimit, itemLimit)
        print("=====================[ RESULT n=", negaCaseNum, " ]=======================")
        print("aucs_dict           = ", aucs_dict)
        print("rmse_dict           = ", rmse_dict)
        print("pos_rmse_dict   = ", pos_rmse_dict)
        print("nega_rmse_dict = ", nega_rmse_dict)
        print("precision_1_dict = ", precision_1_dict)
        print("recall_1_dict       = ", recall_1_dict)
        print("precision_0_dict = ", precision_0_dict)
        print("recall_0_dict       = ", recall_0_dict)

        aucs_pkg.append(aucs_dict)
        rmse_pkg.append(rmse_dict)
        pos_rmse_pkg.append(pos_rmse_dict)
        nega_rmse_pkg.append(nega_rmse_dict)
        precision_1_pkg.append(precision_1_dict)
        precision_0_pkg.append(precision_0_dict)
        recall_1_pkg.append(recall_1_dict)
        recall_0_pkg.append(recall_0_dict)
        print("---------------------------------------------------------")
        print("Evaluation result of [ n="+str(negaCaseNum)+" iter="+str(iterNum)+" data="+fileName+" ] is added to pkg")
        print("---------------------------------------------------------")

    print("[Start printing Graph ... ]")
    printGraph("AUC", iterNum, fileName, aucs_pkg)
    printGraph("RMSE", iterNum, fileName, rmse_pkg)
    printGraph("Positive RMSE", iterNum, fileName, pos_rmse_pkg)
    printGraph("Negative RMSE", iterNum, fileName, nega_rmse_pkg)
    printGraph("Precision (like)", iterNum, fileName, precision_1_pkg)
    printGraph("Precision (dislike)", iterNum, fileName, precision_0_pkg)
    printGraph("Recall (like)", iterNum, fileName, recall_1_pkg)
    printGraph("Recall (dislike)", iterNum, fileName, recall_0_pkg)
    print("[Printing Graph Finished ! ]")


if __name__ == "__main__":
    experiment1_by_negaCaseNum([20000, 50000, 80000], 10, "MovieLens100K_oneClass.csv", 943, 1682)
    


    







