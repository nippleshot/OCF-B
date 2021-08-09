import os
os.chdir("/content/drive/MyDrive/GraduationLab/Graduation/GraduationCode/DataPreprocessing")
import pandas as pd
from tqdm import tqdm
import cupy as cp
import numpy as np
cp.cuda.Device(0).use()

codeDirectory = os.path.dirname(os.getcwd())

def save_zeroClassList(MFDataName, newDataName, userIDLimit, itemIDLimit):
    dataPath = os.path.join(codeDirectory, "Data", MFDataName)
    MF_Data = pd.read_csv(dataPath)
    MF_Data.columns = ["user", "item", "rating", "timestamp"]
    MF_numpy =  MF_Data.to_numpy()
    MF_cupy = cp.asarray(MF_numpy)
    
    newList_data = pd.DataFrame(data=[[0, 0, 0, 100000000]])
    newList_data.columns = ["user", "item", "rating", "timestamp"]
    new_numpy =  newList_data.to_numpy()
    new_cupy = cp.asarray(new_numpy)

    pbar = tqdm(total=userIDLimit*itemIDLimit + 1)
    for userID in range(1, userIDLimit+1):
        for itemID in range(1, itemIDLimit+1):
            isExist = cp.where((MF_cupy[:,0] == userID) & (MF_cupy[:,1] ==  itemID) & (MF_cupy[:,2] == 0))
            if len(isExist[0]) > 0 :
              new_data = cp.array([cp.array(userID), cp.array(itemID), cp.array(0), cp.array(100000000)])
              new_data = cp.reshape(new_data, (1,4))
              new_cupy = cp.concatenate((new_cupy, new_data), axis = 0)
            pbar.update(1)

    pbar.close()
    new_numpy = cp.asnumpy(new_cupy)
    newList_data = pd.DataFrame(new_numpy, columns=["user", "item", "rating", "timestamp"])
    newList_data.sort_values(by=['user', 'item'], inplace=True)

    newDataPath = os.path.join(codeDirectory, "Data", newDataName)
    print("saving ", newDataName, "....")
    newList_data.to_csv(newDataPath, header=False, index=False)
    print("Completed ")


def convert_MFData(oneClassDataName, newDataName, userIDLimit, itemIDLimit):
    dataPath = os.path.join(codeDirectory, "Data", oneClassDataName)
    oneClass_Data = pd.read_csv(dataPath)
    oneClass_Data.columns = ["user", "item", "rating", "timestamp"]
    oneClass_numpy =  oneClass_Data.to_numpy()
    oneClass_cupy = cp.asarray(oneClass_numpy)

    pbar = tqdm(total=userIDLimit*itemIDLimit + 1)
    for userID in range(1, userIDLimit+1):
        for itemID in range(1, itemIDLimit+1):
            isExist = cp.where((oneClass_cupy[:,0] == userID) & (oneClass_cupy[:,1] ==  itemID))
            if len(isExist[0]) < 1 :
              new_data = cp.array([cp.array(userID), cp.array(itemID), cp.array(0), cp.array(100000000)])
              new_data = cp.reshape(new_data, (1,4))
              oneClass_cupy = cp.concatenate((oneClass_cupy, new_data), axis = 0)
            pbar.update(1)

    pbar.close()
    oneClass_numpy = cp.asnumpy(oneClass_cupy)
    oneClass_Data = pd.DataFrame(oneClass_numpy, columns=["user", "item", "rating", "timestamp"])
    oneClass_Data.sort_values(by=['user', 'item'], inplace=True)

    newDataPath = os.path.join(codeDirectory, "Data", newDataName)
    print("saving ", newDataName, "....")
    oneClass_Data.to_csv(newDataPath, header=False, index=False)
    print("Completed ")

if __name__ == "__main__":
    save_zeroClassList("MovieLens100K_MF.csv", "MovieLens100K_ZeroList.csv", 943, 1682)

