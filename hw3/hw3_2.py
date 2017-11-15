import gzip
from collections import defaultdict
import numpy as np
import random
import math

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

data_set = []
u2b = {}
users = set([])
businesses = set([])
for l in readGz("train.json.gz"):
    user,business = l['userID'],l['businessID']
    users.add(user)
    businesses.add(business)
    if user not in u2b:
        u2b[user] = set([business])
    else:
        u2b[user].add(business);
    data_set.append([user, business, 1])

train_set = data_set[:int(len(data_set)/2)]
validate_set = data_set[int(len(data_set)/2):]

# randomly create 100000 negative samples
users = list(users)
businesses = list(businesses)
neg_samples = []
while len(neg_samples) < 100000:
    selected_user = users[int(np.random.rand()*len(users))]
    selected_business = businesses[int(np.random.rand()*len(businesses))]
    if selected_business not in u2b[selected_user]:
        neg_samples.append([selected_user, selected_business, 0])

validate_set = validate_set + neg_samples

# use baseline model to train
businessCount = defaultdict(int)
totalPurchases = 0

i = 1;
for l in readGz("train.json.gz"):
    if i <= 100000:
        user,business = l['userID'],l['businessID']
        businessCount[business] += 1
        totalPurchases += 1
    else:
        break

mostPopular = [(businessCount[x], x) for x in businessCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > 0.618*totalPurchases: break

prediction = []
correct_number = 0
for elem in validate_set:
    if elem[1] in return1:
        prediction.append(1)
        if elem[2] == 1:
            correct_number += 1
    else:
        prediction.append(0)
        if elem[2] == 0:
            correct_number += 1

accuracy = correct_number/len(validate_set)

print(accuracy)
