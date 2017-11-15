import gzip
from collections import defaultdict
import math

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

alpha = 0
beta_u = defaultdict(list)
beta_i = defaultdict(list)
users = set([])
businesses = set([])
u2b = {}
b2u = {}
train_data = []
i = 1
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


users = list(users)
businesses = list(businesses)

def iterate(lamda):
    global alpha
    # update alpha
    numerator = 0
    for d in train_data:
        numerator += (d[2] - beta_u[d[0]] - beta_i[d[1]])
    alpha = numerator/len(train_data)
    # update beta_u
    for user in users:
        total = 0
        for elem in u2b[user]:
            total += (elem[1] - alpha - beta_i[elem[0]])
        beta_u[user] = total/(lamda + len(u2b[user]))
    # update beta_i
    for b in businesses:
        total = 0
        for elem in b2u[b]:
            total += (elem[1] - alpha - beta_u[elem[0]])
        beta_i[b] = total/(lamda + len(b2u[b]))
        # if b == target:
        #     print(beta_i[target])

for x in range(30):
    iterate(10)
# mean_sq_error = 0
# for d in validate_data:
#     mean_sq_error += (alpha + beta_u[d[0]] + beta_i[d[1]] - d[2])**2
# mean_sq_error = math.sqrt(mean_sq_error)
# mean_sq_error = mean_sq_error/100000
# print(mean_sq_error)

idx = 1
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
            predictions.write(u + '-' + i + ',' + str(alpha + beta_u[u] + beta_i[i]) + '\n')
            # if idx < 10:
            #     print(alpha," ", beta_u[u]," ", beta_i[i])
            #     idx+=1

predictions.close()
