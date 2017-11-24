import gzip
from collections import defaultdict
import numpy as np
import random
import math

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

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
        u2b[user].add(business)

users = list(users)

def findneighbour(user, users, u2b, k):
    visited = u2b[user]
    neighbours = []
    for cur_user in users:
        if cur_user == user:
            continue
        else:
            cur_visited = u2b[cur_user]
            intersec = visited.intersection(cur_visited)
            if len(intersec) == 0:
                continue
            else:
                similarity = len(intersec)/math.sqrt(len(visited)*len(cur_visited)) # cosine similarity
                neighbours.append([similarity, cur_user])
    neighbours.sort()
    neighbours.reverse()
    neighbours = neighbours[:k]
    return neighbours

####################### Approach 1 ####################################
# def predict(neighbours, u2b, business):
#     numerator = 0
#     denumerator = 0
#     for i in range(len(neighbours)):
#         user = neighbours[i][1]
#         sim = neighbours[i][0]
#         denumerator += sim
#         if business in u2b[user]:
#             numerator += sim
#     if numerator/denumerator > 0.5:
#         return True
#     else:
#         return False
#
# predictions = open("predictions_Visit.txt", 'w')
# for l in open("pairs_Visit.txt"):
#     if l.startswith("userID"):
#         #header
#         predictions.write(l)
#         continue
#     u,i = l.strip().split('-')
#     if u in u2b:
#         neighbours = findneighbour(u, users, u2b, 5) # when set 50, the score is 0.75380
#         if predict(neighbours, u2b, i):
#             predictions.write(u + '-' + i + ",1\n")
#         else:
#             predictions.write(u + '-' + i + ",0\n")
#     else:
#         predictions.write(u + '-' + i + ",0\n")
#
# predictions.close()

####################### Approach 2 ####################################
def predict(neighbours, u2b, business):
    for i in range(len(neighbours)):
        user = neighbours[i][1]
        if business in u2b[user]:
            return True
    return False

predictions = open("predictions_Visit.txt", 'w')
for l in open("pairs_Visit.txt"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,i = l.strip().split('-')
    if u in u2b:
        neighbours = findneighbour(u, users, u2b, 500) # when set 50, the score is 0.75380
        if predict(neighbours, u2b, i):
            predictions.write(u + '-' + i + ",1\n")
        else:
            predictions.write(u + '-' + i + ",0\n")
    else:
        predictions.write(u + '-' + i + ",0\n")

predictions.close()
