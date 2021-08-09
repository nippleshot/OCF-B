import os
import pandas as pd

"""
<Epinions> DATA INFO : 

The dataset contains
* 49,290 users who rated a total of
* 139,738 different items at least once, writing
* 664,824 reviews and
* 487,181 issued trust statements.

Every line has the following format: 
user_id | item_id | rating_value

Ranges:
user_id is in [1,49290]
item_id is in [1,139738]
rating_value is in [1,5]
"""

codeDirectory = os.path.dirname(os.getcwd())
rawDataPath = os.path.join(codeDirectory,"Data","Raw","epinions_ratings_data.txt")

def review_fullData(full_Data):
    print("[Data Total] : ", len(full_Data))

    full_Data["item"] = full_Data.item.astype('category')
    print("[item INFO] : \n", full_Data["item"].unique())
    print("\n")

    full_Data["user"] = full_Data.user.astype('category')
    print("[user INFO] : \n", full_Data["user"].unique())
    print("\n")


    grouped_rating = full_Data.groupby("rating")
    print("[rating INFO] : \n", grouped_rating.size())
    print("\n")

def save_Data(data, filename):
    newDataPath = os.path.join(codeDirectory, "Data", filename)
    f = open(newDataPath, 'w')
    for element in data:
        f.write(element+'\n')
    f.close()

def convert_oneClass():
    new_data = []
    f = open(rawDataPath, 'r')
    lines = f.readlines()
    for line in lines:
        line_format = line.split( );
        if line_format[2] == '5' or line_format[2] == '4':
            new_data.append(str(line_format[0])+","+str(line_format[1])+","+"1"+","+"111111111")
    f.close()
    print("--- convert_oneClass() ---")
    print("number of positive data : ", len(new_data))
    print("\n")

    return new_data

def convert_twoClass():
    new_data = []
    pos_count = 0
    neg_count = 0

    f = open(rawDataPath, 'r')
    lines = f.readlines()
    for line in lines:
        line_format = line.split();
        if line_format[2] == '5' or line_format[2] == '4':
            new_data.append(str(line_format[0]) + "," + str(line_format[1]) + "," + "1" + "," + "111111111")
            pos_count = pos_count + 1
        elif line_format[2] == '2' or line_format[2] == '1':
            new_data.append(str(line_format[0]) + "," + str(line_format[1]) + "," + "0" + "," + "111111111")
            neg_count = neg_count + 1
    f.close()
    print("--- convert_twoClass() ---")
    print("number of total data : ", len(new_data))
    print("positive data : {} -- ({:.1%})".format(pos_count, pos_count/len(new_data)))
    print("negative data : {} -- ({:.1%})".format(neg_count, neg_count/len(new_data)))
    print("\n")

    return new_data

if __name__ == "__main__":
    # save_Data(convert_oneClass(), "Epinions_oneClass.csv")
    # save_Data(convert_twoClass(), "Epinions_twoClass.csv")

    # raw_data = pd.read_csv(rawDataPath, header=None, sep=' ')
    # raw_data.columns = ["user", "item", "rating"]
    # review_fullData(raw_data)

    newDataPath = os.path.join(codeDirectory, "Data", "Epinions_oneClass.csv")
    new_data = pd.read_csv(newDataPath)
    new_data.columns = ["user", "item", "rating", "timestamp"]
    review_fullData(new_data)

