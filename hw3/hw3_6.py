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
validate_data = []
i = 1
for l in readGz("train.json.gz"):
    user,business,rate = l['userID'],l['businessID'],l['rating']
    if i <= 100000:
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
    else:
        validate_data.append([user, business, rate])
    i+=1
    beta_u[user] = 0
    beta_i[business] = 0

users = list(users)
businesses = list(businesses)
target = businesses[0]

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


for x in range(20):
    iterate(10)

# validate
mean_sq_error = 0
for d in validate_data:
    mean_sq_error += (alpha + beta_u[d[0]] + beta_i[d[1]] - d[2])**2
mean_sq_error = mean_sq_error/100000
print("MSE is equal to ", mean_sq_error)

# question 7

# beta_max_u = 0
# beta_max_i = 0
# max_u = []
# max_i = []
#
# beta_min_u = 0
# beta_min_i = 0
# min_u = []
# min_i = []
#
# for user in users:
#     if beta_u[user] > beta_max_u:
#         max_u = [user]
#         beta_max_u = beta_u[user]
#     elif beta_u[user] == beta_max_u:
#         max_u.append(user)
#     if beta_u[user] < beta_min_u:
#         min_u = [user]
#         beta_min_u = beta_u[user]
#     elif beta_u[user] == beta_min_u:
#         min_u.append(user)
#
# for b in businesses:
#     if beta_i[b] > beta_max_i:
#         max_i = [b]
#         beta_max_i = beta_i[b]
#     elif beta_i[b] == beta_max_i:
#         max_i.append(b)
#     if beta_i[b] < beta_min_i:
#         min_i = [b]
#         beta_min_i = beta_i[b]
#     elif beta_i[b] == beta_min_i:
#         min_i.append(b)
#
# print(max_u, beta_max_u)
# print(max_i, beta_max_i)
# print(min_u, beta_min_u)
# print(min_i, beta_min_i)
