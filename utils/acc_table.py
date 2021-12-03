import random
import pandas as pd

#leftside label
label = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


acc1 = [] # column 1
acc2 = [] # column 2

for i in range(10):
    acc1.append(str(round(random.uniform(0,100), 2)))
    acc2.append(str(round(random.uniform(0,100), 2)))
    

final_list = [acc1, acc2] # save two list in a list as a nested form


df = pd.DataFrame(final_list) # create dataframe
df = df.T # transpose it
df.insert(loc=0, column="label", value=label) # add label list as label to 0th column
df.columns = ['label', 'MA', 'AA'] # define top rows 

print(df)
savefile = "C:/Users/seacl/Desktop/test.csv" # filename
df.to_csv(savefile, encoding="utf_8_sig") # save file

del df # release memory
