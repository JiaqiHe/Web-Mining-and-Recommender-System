import gzip
from collections import defaultdict
import math

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

allRatings = []
userRatings = defaultdict(list)
validate_data = []
i = 1
for l in readGz("train.json.gz"):
    if i <= 100000:
        user,business = l['userID'],l['businessID']
        allRatings.append(l['rating'])
        userRatings[user].append(l['rating'])
    else:
        user,business,rate = l['userID'],l['businessID'],l['rating']
        validate_data.append([user, business, rate])
    i+=1

globalAverage = sum(allRatings) / len(allRatings)

print(globalAverage)

MSE = 0
for d in validate_data:
    MSE += (d[2]-globalAverage)**2
MSE /= 100000
print(MSE)
