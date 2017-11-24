import gzip
from collections import defaultdict
import math

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

alpha = 4.2
beta_u = defaultdict(list)
beta_i = defaultdict(list)
users = set([])
businesses = set([])
u2b = {}
b2u = {}
train_data = []
for l in readGz("train.json.gz"):
    user,business,rate = l['userID'],l['businessID'],l['rating']
    beta_u[user] = 0
    beta_i[business] = 0
    train_data.append([user, business, rate])
    if user not in u2b:
        u2b[user] = set([(business,rate)])
        users.add(user)
    else:
        u2b[user].add((business, rate))
    if business not in b2u:
        b2u[business] = set([(user,rate)])
        businesses.add(business)
    else:
        b2u[business].add((user,rate))


# matrix factorization
import numpy as np
k = 1
user_map = {}
business_map = {}
idx_user = 0           #the number of different users
idx_business = 0       #the number of different businesses
for user in users:
    user_map[user] = idx_user
    idx_user += 1
for business in businesses:
    business_map[business] = idx_business
    idx_business += 1
p = np.random.rand(idx_user, k)*math.sqrt(5.0/k)     # user
q = np.random.rand(idx_business, k)*math.sqrt(5.0/k) # business

def iterate(lamda):
    global alpha
    # update alpha
    numerator = 0
    for d in train_data:
        u = user_map[d[0]]
        i = business_map[d[1]]
        numerator += (d[2] - beta_u[d[0]] - beta_i[d[1]] - np.dot(p[u,],q[i,]))
    alpha = numerator/len(train_data)
    # update beta_u
    for user in users:
        u = user_map[user]
        total = 0
        for elem in u2b[user]:
            i = business_map[elem[0]]
            total += (elem[1] - alpha - beta_i[elem[0]] - np.dot(p[u,],q[i,]))
        beta_u[user] = total/(lamda + len(u2b[user]))
    # update beta_i
    for b in businesses:
        i = business_map[b]
        total = 0
        for elem in b2u[b]:
            u = user_map[elem[0]]
            total += (elem[1] - alpha - beta_u[elem[0]] - np.dot(p[u,],q[i,]))
        beta_i[b] = total/(lamda + len(b2u[b]))
    # update p_u
    for user in users:
        u = user_map[user]
        temp_p = [0] * k
        temp_p = np.array(temp_p)
        denominator = 0
        for elem in u2b[user]:
            i = business_map[elem[0]]
            temp_p = (elem[1] - alpha - beta_u[user] - beta_i[elem[0]])*q[i,] + temp_p
            denominator += sum(q[i,]**2)
        denominator += lamda
        p[u,] = temp_p/denominator
    # update q_i
    for b in businesses:
        i = business_map[b]
        temp_q = [0] * k
        temp_q = np.array(temp_q)
        denominator = 0
        for elem in b2u[b]:
            u = user_map[elem[0]]
            temp_q = (elem[1] - alpha - beta_u[elem[0]] - beta_i[b])*p[u,] + temp_q
            denominator += sum(p[u,]**2)
        denominator += lamda
        q[i,] = temp_q/denominator

for x in range(30):
    iterate(7)
    if x%10 == 0:
        print(alpha)

error = 0
for l in readGz("train.json.gz"):
    user,business,r = l['userID'],l['businessID'],l['rating']
    u = user_map[user]
    i = business_map[business]
    error += (r-np.dot(q[i,],p[u,])-alpha-beta_u[user]-beta_i[business])**2
print(error/200000)

predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,i = l.strip().split('-')
    if u not in users:
        if i not in businesses:
            predictions.write(u + '-' + i + ',' + str(alpha) + '\n')
        else:
            predictions.write(u + '-' + i + ',' + str(alpha + beta_i[i]) + '\n')
    else:
        if i not in businesses:
            predictions.write(u + '-' + i + ',' + str(alpha + beta_u[u]) + '\n')
        else:
            u_idx = user_map[u]
            i_idx = business_map[i]
            predicted_rating = alpha + beta_u[u] + beta_i[i] + np.dot(p[u_idx,],q[i_idx,])
            if predicted_rating > 5.0:
                predicted_rating = 5.0
            predictions.write(u + '-' + i + ',' + str(predicted_rating) + '\n')

predictions.close()
