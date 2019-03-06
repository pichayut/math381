import csv
import pandas as pd
import pickle
import random
import numpy as np
from statistics import *

"""
# Get city names in Washington state
"""
df = pd.read_csv('seattleWeather_1948-2017.csv', encoding = "ISO-8859-1")

year = []
month = []
day = []
datenum = []
for i in range(len(df)):
    strr = df.iloc[i]['DATE']
    date = strr.split('-')
    #print(date)
    year.append(date[0])
    month.append(date[1])
    day.append(date[2])
    datenum.append(int(date[1]) * 31 + int(date[2]))

df['YEAR'] = year
df['MONTH'] = month
df['DAY'] = day
df['DATENUM'] = datenum

df = df[(df['YEAR'] >= "2007") & (df['YEAR'] <= "2017")] # 2007-2017
df = df.assign(MEAN=(df['TMAX'] + df['TMIN']) / 2)

"""
df = df.assign(INTERVAL=df['MEAN'] // 10)
"""
avg = []
for i in range(len(df)):
    avg.append(df.iloc[i]['MEAN'])
mu = mean(avg)
sd = stdev(avg) / 2

# Winter
df = df[(df['DATENUM'] >= 393) | (df['DATENUM'] <= 113)] # 21 Dec - Mar 20

# Spring
# df = df[(df['DATENUM'] <= 6 * 31 + 21) & (df['DATENUM'] >= 113)] # 20 Mar - June 21

# Summer
#df = df[(df['DATENUM'] >= 6 * 31 + 21) & (df['DATENUM'] <= 31 * 9 + 23)] # 21 June - Sep 23

# Fall
#df = df[(df['DATENUM'] >= 9 * 31 + 23) & (df['DATENUM'] <= 31 * 12 + 21)] # Sep 23 - Dec 21

interval = []
for i in range(len(df)):
    val = df.iloc[i]['MEAN']
    for j in range(-10, 10):
        if mu + j * sd <= val and val <= mu + (j+1) * sd:
            interval.append(j)
            break
        
df['INTERVAL'] = interval

for i in range(-10, 10):
    print(i, ": (", mu + i *sd, ", ", mu + (i+1) * sd, ")")

matrix = dict()
entry = set()
for i in range(len(df)-1):
    entry.add((df.iloc[i]['INTERVAL'],df.iloc[i]['RAIN']))
    if i == len(df)-2:
        entry.add((df.iloc[i+1]['INTERVAL'],df.iloc[i+1]['RAIN']))
    key = ((df.iloc[i]['INTERVAL'],df.iloc[i]['RAIN']),(df.iloc[i+1]['INTERVAL'],df.iloc[i+1]['RAIN']))
    if key not in matrix:
        matrix[key] = 1
    else:
        matrix[key] += 1

entry = list(entry)
entry.sort()
with open("matrix_mu_sd_10yrs_winter.tsv", 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(["today/tmr"] + entry)
    for i in range(len(entry)):
        next_pos = entry[i]
        row = []
        for j in range(len(entry)):
            cur_pos = entry[j]
            if (cur_pos, next_pos) in matrix:
                row += [matrix[(cur_pos, next_pos)]]
            else:
                row += [0]
        row=[float(i)/sum(row) for i in row]
        row.insert(0,next_pos)
        tsv_writer.writerow(row)
        
###### 2 previous days
        
matrix = dict()
entry = set()
for i in range(len(df)-2):
    entry.add((df.iloc[i]['INTERVAL'],df.iloc[i]['RAIN']))
    entry.add((df.iloc[i]['INTERVAL'],df.iloc[i]['RAIN'],df.iloc[i+1]['INTERVAL'],df.iloc[i+1]['RAIN']))
    if i == len(df)-3:
        entry.add((df.iloc[i+1]['INTERVAL'],df.iloc[i+1]['RAIN']))
        entry.add((df.iloc[i+2]['INTERVAL'],df.iloc[i+2]['RAIN']))
    key = ((df.iloc[i]['INTERVAL'],df.iloc[i]['RAIN'],df.iloc[i+1]['INTERVAL'],df.iloc[i+1]['RAIN']),(df.iloc[i+2]['INTERVAL'],df.iloc[i+2]['RAIN']))
    if key not in matrix:
        matrix[key] = 1
    else:
        matrix[key] += 1

entry = list(entry)
entry.sort()
with open("matrix_mu_sd_10yrs_winter_2days.tsv", 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    lst = list()
    for j in range(len(entry)):
        if len(entry[j]) == 2:
            lst.append(entry[j])
    tsv_writer.writerow(["ytd-today/tmr"] + lst)
    for i in range(len(entry)):
        cur_pos = entry[i]
        row = []
        for j in range(len(entry)):
            next_pos = entry[j]
            if len(next_pos) == 2:
                if (cur_pos, next_pos) in matrix:
                    print (cur_pos, next_pos)
                    row += [matrix[(cur_pos, next_pos)]]
                else:
                    row += [0]
        if sum(row) != 0:
            row=[float(i)/sum(row) for i in row]
            row.insert(0,cur_pos)
            tsv_writer.writerow(row)
        
#place = set(df['place'])
#pickle_out = open("./data/code.pickle","wb")
#pickle.dump(place, pickle_out)
#pickle_out.close()
        
print ("\n")

"""
# Double Rain
double_rain_freq = 0
total = 0
for i in range(len(df) - 1):
    if df.iloc[i]["YEAR"] == df.iloc[i+1]["YEAR"]:
        if df.iloc[i]["RAIN"] == True and df.iloc[i+1]["RAIN"] == True:
            double_rain_freq += 1
        total += 1
double_rain_freq /= total
print ("Double Rain Freq:", double_rain_freq)

# Triple Rain
triple_rain_freq = 0
total = 0
for i in range(len(df) - 2):
    if df.iloc[i]["YEAR"] == df.iloc[i+1]["YEAR"] and df.iloc[i+1]["YEAR"] == df.iloc[i+2]["YEAR"]:
        if df.iloc[i]["RAIN"] == True and df.iloc[i+1]["RAIN"] == True and df.iloc[i+2]["RAIN"] == True:
            triple_rain_freq += 1
        total += 1
triple_rain_freq /= (len(df) - 2)
print ("Triple Rain Freq:", triple_rain_freq)

# Double No-rain
double_norain_freq = 0
total = 0
for i in range(len(df) - 1):
    if df.iloc[i]["YEAR"] == df.iloc[i+1]["YEAR"]:
        if df.iloc[i]["RAIN"] == False and df.iloc[i+1]["RAIN"] == False:
            double_norain_freq += 1
        total += 1
double_norain_freq /= total
print ("Double No-rain Freq:", double_norain_freq)

# Triple No-rain
triple_norain_freq = 0
total = 0
for i in range(len(df) - 2):
    if df.iloc[i]["YEAR"] == df.iloc[i+1]["YEAR"] and df.iloc[i+1]["YEAR"] == df.iloc[i+2]["YEAR"]:
        if df.iloc[i]["RAIN"] == False and df.iloc[i+1]["RAIN"] == False and df.iloc[i+2]["RAIN"] == False:
            triple_norain_freq += 1
        total += 1
triple_norain_freq /= total
print ("Triple No-rain Freq:", triple_norain_freq)

# No-rain Rain
norain_rain_freq = 0
total = 0
for i in range(len(df) - 1):
    if df.iloc[i]["YEAR"] == df.iloc[i+1]["YEAR"]:
        if df.iloc[i]["RAIN"] == False and df.iloc[i+1]["RAIN"] == True:
            norain_rain_freq += 1
        total += 1
norain_rain_freq /= total
print ("No-rain Rain Freq:", norain_rain_freq)

# Rain No-rain 
rain_norian_freq = 0
total = 0
for i in range(len(df) - 1):
    if df.iloc[i]["YEAR"] == df.iloc[i+1]["YEAR"]:
        if df.iloc[i]["RAIN"] == True and df.iloc[i+1]["RAIN"] == False:
            rain_norian_freq += 1
        total += 1
rain_norian_freq /= total
print ("Rain No-rain Freq:", rain_norian_freq)

# No-rain Rain No-rain
nr_r_nr_freq = 0
total = 0
for i in range(len(df) - 2):
    if df.iloc[i]["YEAR"] == df.iloc[i+1]["YEAR"] and df.iloc[i+1]["YEAR"] == df.iloc[i+2]["YEAR"]:
        if df.iloc[i]["RAIN"] == False and df.iloc[i+1]["RAIN"] == True and df.iloc[i+2]["RAIN"] == False:
            nr_r_nr_freq += 1
        total += 1
nr_r_nr_freq /= total
print ("Nr R Nr Freq:", nr_r_nr_freq)

# Rain no-rain Rain
r_nr_r_freq = 0
total = 0
for i in range(len(df) - 2):
    if df.iloc[i]["YEAR"] == df.iloc[i+1]["YEAR"] and df.iloc[i+1]["YEAR"] == df.iloc[i+2]["YEAR"]:
        if df.iloc[i]["RAIN"] == True and df.iloc[i+1]["RAIN"] == False and df.iloc[i+2]["RAIN"] == True:
            r_nr_r_freq += 1
        total += 1
r_nr_r_freq /= total
print ("R Nr R Freq:", r_nr_r_freq)

######################################

# Stays still
SS = np.zeros((7, 8))
for interval in range(0, -7, -1):
    for freq in range(2, 10):
        print(interval, freq)
        total = 0
        cnt = 0
        for st in range(len(df) - (freq - 1)):
            same_year = True
            for i in range(st, st + freq - 1):
                if df.iloc[i]["YEAR"] != df.iloc[i+1]["YEAR"]:
                    same_year = False
                    break
            if same_year == True:
                flag = True
                for i in range(st, st + freq - 1):
                    if df.iloc[i]["INTERVAL"] != interval:
                        flag = False
                        break
                if flag is True:
                    cnt += 1
            total += 1
    SS[abs(interval)][freq - 2] = cnt/total
"""
    
"""
### FASTER
tot = np.zeros((8))
for freq in range(2, 10):
    total = 0
    for st in range(len(df) - (freq - 1)):
        same_year = True
        if df.iloc[st]["YEAR"] == df.iloc[st + freq - 2]["YEAR"]:
            tot[freq - 2] += 1

SS = np.zeros((7 + 1, 8 + 1))
for st in range(len(df)):
    base = df.iloc[st]["INTERVAL"]
    freq = 1
    for i in range(st + 1, st + 9):
        if i >= len(df): 
            break
        if df.iloc[i]["YEAR"] != df.iloc[st]["YEAR"]:
            break
        cur = df.iloc[i]["INTERVAL"]
        if cur != base: 
            break
        freq += 1
        SS[abs(base) + 1][freq - 2 + 1] += (1/tot[freq-2])

SS = np.round_(SS*100, 2)

for i in range(0, -7, -1):
    SS[abs(i) + 1][0] = i
    
for i in range(2, 10):
    SS[0][i - 1] = i

print (SS)
np.savetxt("stay-still.csv", SS, delimiter=",", fmt='%.2f')
"""

### I 2, 3, 4 in a row
J = [0,0,0]
for st in range(len(df)):
    prev = df.iloc[st]["INTERVAL"]
    freq = 1
    for i in range(st + 1, st + 4):
        if i >= len(df): 
            break
        if df.iloc[i]["YEAR"] != df.iloc[st]["YEAR"]:
            break
        cur = df.iloc[i]["INTERVAL"]
        if cur <= prev: 
            break
        freq += 1
        prev = cur
        J[freq - 2] += 1
print ("INCREASE 2,3,4", J[0]/tot[0] * 100, J[1]/tot[1] * 100, J[2]/tot[2] * 100)

### D 2, 3, 4 in a row
J = [0,0,0]
for st in range(len(df)):
    prev = df.iloc[st]["INTERVAL"]
    freq = 1
    for i in range(st + 1, st + 4):
        if i >= len(df): 
            break
        if df.iloc[i]["YEAR"] != df.iloc[st]["YEAR"]:
            break
        cur = df.iloc[i]["INTERVAL"]
        if cur >= prev: 
            break
        freq += 1
        prev = cur
        J[freq - 2] += 1
print ("DECREASE 2,3,4", J[0]/tot[0] * 100, J[1]/tot[1] * 100, J[2]/tot[2] * 100)

### JUMP 2, 3, 4 in a row
J = [0,0,0]
for st in range(len(df)):
    prev = df.iloc[st]["INTERVAL"]
    freq = 1
    for i in range(st + 1, st + 4):
        if i >= len(df): 
            break
        if df.iloc[i]["YEAR"] != df.iloc[st]["YEAR"]:
            break
        cur = df.iloc[i]["INTERVAL"]
        if cur == prev: 
            break
        freq += 1
        prev = cur
        J[freq - 2] += 1
print ("JUMP 2,3,4", J[0]/tot[0], J[1]/tot[1], J[2]/tot[2])

### UDU 2, 3, 4 in a row
J = 0
for st in range(len(df)):
    prev = df.iloc[st]["INTERVAL"]
    freq = 1
    for i in range(st + 1, st + 4):
        if i >= len(df): 
            break
        if df.iloc[i]["YEAR"] != df.iloc[st]["YEAR"]:
            break
        cur = df.iloc[i]["INTERVAL"]
        if (cur <= prev and freq % 2 == 1) or (cur >= prev and freq % 2 == 0): 
            break
        prev = cur
        J += 1
print ("U D U", J/tot[2])

### DUD 2, 3, 4 in a row
J = 0
for st in range(len(df)):
    prev = df.iloc[st]["INTERVAL"]
    freq = 1
    for i in range(st + 1, st + 4):
        if i >= len(df): 
            break
        if df.iloc[i]["YEAR"] != df.iloc[st]["YEAR"]:
            break
        cur = df.iloc[i]["INTERVAL"]
        if (cur >= prev and freq % 2 == 1) or (cur <= prev and freq % 2 == 0): 
            break
        prev = cur
        J += 1
print ("D U D", J/tot[2])