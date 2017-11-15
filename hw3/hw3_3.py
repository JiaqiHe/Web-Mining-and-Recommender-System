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
u2t = {}
b2t = {}
i = 1
for l in readGz("train.json.gz"):
    user,business,types = l['userID'],l['businessID'],l['categories']
    users.add(user)
    businesses.add(business)
    if user not in u2b:
        u2b[user] = set([business])
    else:
        u2b[user].add(business);
    data_set.append([user, business, 1])
    if i <=100000:
        i += 1
        # update user -> types that they visit
        if user not in u2t:
            u2t[user] = set(types)
        else:
            for t in types:
                u2t[user].add(t)
        # update business -> types that they belong to
        if business not in b2t:
            b2t[business] = set(types)
        else:
            for t in types:
                b2t[business].add(t)


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

# predict the results
prediction = []
correct_number = 0
for elem in validate_set:
    if elem[0] in u2t and elem[1] in b2t:
        types_user = u2t[elem[0]]
        types_business = b2t[elem[1]]
        size1 = len(types_user)
        size2 = len(types_business)
        types_user = types_user.union(types_business)
        size3 = len(types_user)
        if size1 + size2 != size3:
            prediction.append(1)
            if elem[2] == 1:
                correct_number += 1
        else:
            prediction.append(0)
            if elem[2] == 0:
                correct_number += 1
    else:
        prediction.append(0)
        if elem[2] == 0:
            correct_number += 1

accuracy = correct_number/len(validate_set)

print(accuracy)
