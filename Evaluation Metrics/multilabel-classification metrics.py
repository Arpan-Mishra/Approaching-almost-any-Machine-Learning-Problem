# - Precision @ k
# - Average Precision @ k
# - mean average precision @ k
# - log loss


def pk(y_true,y_pred,k):
    """ Calculated precision at k for a single sample
    y_true - list of values, actual classes
    y_pred - list of values, predicted classes,
    returns: precision at a given value k"""
    # if k = 0, return 0
    # k always >=1
    
    if k==0:
        return 0
    else:
        y_pred = y_pred[:k]
        # convert predictions to set
        pred_set = set(y_pred)
        # convert actual vals to set
        true_set = set(y_true)
        
        # common values
        common_values = pred_set.intersection(true_set)
        # return length of common values over k
        
        return len(common_values)/len(y_pred[:k])
    
def apk(y_true,y_pred,k):
    if k==0:
        return 0
    else:
        pk_values = []
        for i in range(1,k+1):
            pk_values.append(pk(y_true,y_pred,i))
            
    if len(pk_values) == 0:
        return 0
    else:
        return sum(pk_values)/len(pk_values)

y_true = [
        [1,2,3],
        [0,2],
        [1],
        [2,3],
        [1,0],
        []
        ]
y_pred = [
        [0,1,2],
        [1],
        [0,2,3],
        [2,3,4,0],
        [0,1,2],
        [0]
        ]

for i in range(len(y_true)):
    for j in range(1,4):
        print(
                f"""
                y_true = {y_true[i]},
                y_pred = {y_pred[i]},
                AP@{j} = {apk(y_true[i],y_pred[i],k=j)}
                """
                )

# mean apk@k
        
def mapk(y_true,y_pred,k):
    apk_values = []
    for i in range(len(y_true)):
        apk_values.append(apk(y_true[i],y_pred[i],k))
    return sum(apk_values)/len(apk_values)


mapk(y_true,y_pred,k = 1)
mapk(y_true,y_pred,k = 2)
mapk(y_true,y_pred,k = 3)
mapk(y_true,y_pred,k = 4)

# log-loss: take average of log loss of each column (label)

# Quadratic Weighted Kappa (or cohen's kappa) - tells the agreement between
# 2 "ratings"

from sklearn import metrics

metrics.cohen_kappa_score?

# MCC - Mathew's Correlation Coefficient

# MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(FN+TN)*TP+FN)












    