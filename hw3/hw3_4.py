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
for l in readGz("train.json.gz"):
    user,business,types = l['userID'],l['businessID'],l['categories']
    users.add(user)
    businesses.add(business)
    if user not in u2b:
        u2b[user] = set([business])
    else:
        u2b[user].add(business);
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
    data_set.append([user, business, 1])

print (len(users))
print (len(businesses))
# predict the results
predictions = open("predictions_Visit.txt", 'w')
for l in open("pairs_Visit.txt"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,i = l.strip().split('-')
    if u in u2t and i in b2t:
        types_user = u2t[u]
        types_business = b2t[i]
        size1 = len(types_user)
        size2 = len(types_business)
        types_user = types_user.union(types_business)
        size3 = len(types_user)
        if size1 + size2 != size3:
            predictions.write(u + '-' + i + ",1\n")
        else:
            predictions.write(u + '-' + i + ",0\n")
    else:
        predictions.write(u + '-' + i + ",0\n")

predictions.close()
