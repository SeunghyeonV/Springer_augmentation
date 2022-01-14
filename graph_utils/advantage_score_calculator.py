import pandas as pd
import csv

file = open("E:/research/research_log/cifar100_advantage/MA.csv", encoding='utf-8-sig')
csvreader = csv.reader(file)

MA_rows = []
for row in csvreader:
        MA_rows.append(row)


file = open("E:/research/research_log/cifar100_advantage/AA.csv", encoding='utf-8-sig')
csvreader = csv.reader(file)

AA_rows = []
for row in csvreader:
        AA_rows.append(row)
        

file = open("E:/research/research_log/cifar100_advantage/DP_MA.csv", encoding='utf-8-sig')
csvreader = csv.reader(file)

DP_MA_rows = []
for row in csvreader:
        DP_MA_rows.append(row)


file = open("E:/research/research_log/cifar100_advantage/DP_AA.csv", encoding='utf-8-sig')
csvreader = csv.reader(file)

DP_AA_rows = []
for row in csvreader:
        DP_AA_rows.append(row)
   

for i in range(10):
    for j in range(10):
        MA_rows[i][j] = float(MA_rows[i][j])
        AA_rows[i][j] = float(AA_rows[i][j])
        DP_MA_rows[i][j] = float(DP_MA_rows[i][j])
        DP_AA_rows[i][j] = float(DP_AA_rows[i][j])



MA_advantage = [[] for _ in range(10)]
AA_advantage = [[] for _ in range(10)]

for i in range(10):
    for j in range(10):
        MA_advantage[i].append(round(MA_rows[i][j] - DP_MA_rows[i][j] , 2))
        AA_advantage[i].append(round(DP_AA_rows[i][j] - AA_rows[i][j] , 2))

print(MA_rows)
print(" ")

print(DP_MA_rows)
print(" ")

print(MA_advantage)
print(" ")

print(DP_AA_rows)
print(" ")

print(AA_rows)
print(" ")

print(AA_advantage)
print(" ")

total_advantage = [[] for _ in range(10)]
for i in range(10):
    for j in range(10):
        total_advantage[i].append(round(MA_advantage[i][j] + AA_advantage[i][j], 2))
        
print(total_advantage)
