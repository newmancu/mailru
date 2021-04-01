import numpy as np

# support function
def binarization(Y, values_dict=None):
    if values_dict is None:
        values_dict = {key:val for val, key in enumerate(set(Y))}
    Y_new = np.array([values_dict[y] for y in Y], dtype=np.int32)
    return (Y_new, values_dict)

def get_samples(y_true, y_predict, percent):
    assert (0 <= percent <= 100), "percent must be in range of 0..100"
    if percent is None:
        percent = 50
    
    Y_true, values_dict = binarization(y_true)
    unique, counts = np.unique(Y_true, return_counts=True)
    frequencies = counts / np.sum(counts)
    N = y_true.shape[0] * percent / 100
    number_of_items = np.array(np.floor(N*frequencies), dtype=np.int32)
    indexes = np.arange(0, Y_true.shape[0], dtype=np.int32)
    categ_iindexes_mas = []
    for i,val in enumerate(values_dict):
        categ_iindexes_mas = np.concatenate((categ_iindexes_mas,
            np.random.choice(indexes[Y_true == val], number_of_items[i], replace=False)
            ))
    categ_iindexes_mas = np.array(categ_iindexes_mas, dtype=np.int32)
    return y_true[categ_iindexes_mas], y_predict[categ_iindexes_mas]

def get_samples_bin(y_true, y_predict, percent):
    assert (0 <= percent <= 100), "percent must be in range of 0..100"
    if percent is None:
        percent = 50
    y_true_b = np.array(y_true, dtype=bool)
    counts = np.array(y_true.sum(axis=0), dtype=np.int32).flatten()
    frequencies = counts / np.sum(counts)
    N = y_true.shape[0] * percent / 100
    number_of_items = np.array(np.floor(N*frequencies), dtype=np.int32)
    indexes = np.arange(0, y_true.shape[0], dtype=np.int32)
    
    categ_indexes_mas = np.array([], dtype=np.int32)
    for i in range(0, y_true.shape[1]):
        categ_indexes_mas = np.concatenate((categ_indexes_mas,
            np.random.choice(indexes[y_true_b.T[i]], number_of_items[i], replace=False)
            ))

    return y_true[categ_indexes_mas], y_predict[categ_indexes_mas]

def get_params(y_true, y_predict):
    y_true_b = np.array(y_true, dtype=bool)
    y_predict_b = np.array(y_predict, dtype=bool)
    
    TP = (y_predict_b & (y_true_b == y_predict_b)).sum(axis=0)
    TN = ((np.logical_not(y_predict_b)) & (y_true_b == y_predict_b)).sum(axis=0)
    FP = (y_predict_b & (y_true_b != y_predict_b)).sum(axis=0)
    FN = ((np.logical_not(y_predict_b)) & (y_true_b != y_predict_b)).sum(axis=0)
    
    return (TP, TN, FP, FN)
    
def jaccard_score(y_true, y_predict, percent=None):
    y_true, y_predict = get_samples_bin(y_true, y_predict, percent)
    TP, TN, FP, FN = get_params(y_true, y_predict)
    return (TP) / (TP + FP + FN)



# main functions
def accuracy_score(y_true, y_predict, percent=None):
    y_true, y_predict = get_samples_bin(y_true, y_predict, percent)
    return np.mean((y_true == y_predict).all(axis=1))
    
def precision_score(y_true, y_predict, percent=None):
    y_true, y_predict = get_samples_bin(y_true, y_predict, percent)
    TP, TN, FP, FN = get_params(y_true, y_predict)
    return TP / (TP + FP)

def recall_score(y_true, y_predict, percent=None):
    y_true, y_predict = get_samples_bin(y_true, y_predict, percent)
    TP, TN, FP, FN = get_params(y_true, y_predict)
    return TP / (TP + FN)

def lift_score(y_true, y_predict, percent=None):
    y_true, y_predict = get_samples_bin(y_true, y_predict, percent)
    TP, TN, FP, FN = get_params(y_true, y_predict)
    return (TP / (TP + FN)) / ((TP + FP) / (TP + TN + FP + FN))

def f1_score(y_true, y_predict, percent=None):
    y_true, y_predict = get_samples_bin(y_true, y_predict, percent)
    TP, TN, FP, FN = get_params(y_true, y_predict)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    res = 2 * (precision * recall)
    indexes = (precision + recall) == 0
    res[indexes] = 0
    res[~indexes] = res[~indexes] / (precision + recall)[~indexes]

    return res

def confusion_matrix(y_true, y_predict, percent=None):
    y_true, y_predict = get_samples_bin(y_true, y_predict, percent)
    TP, TN, FP, FN = get_params(y_true, y_predict)

    y_true_b = np.array(y_true, dtype=bool)
    y_predict_b = np.array(y_predict, dtype=bool)
    A = np.array(y_true_b == y_predict_b, dtype=np.int32)
    matrix = y_true.T@y_predict
    print(TP)
    print(TN)
    print(FP)
    print(FN)
    
    print(matrix)
    

if __name__ == '__main__':
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import classification_report
    from sklearn.metrics import precision_score as prec
    from sklearn.metrics import recall_score as recall
    from sklearn.metrics import f1_score as f1
    from sklearn.metrics import accuracy_score as sk_acc
    from sklearn.metrics import jaccard_score as sk_jacc
    
    
    np.random.seed(1)
    k = 3
    N = 2000
    y_true = np.random.randint(0, k, N)
    y_pred = np.random.randint(0, k, N)
    ohe = OneHotEncoder().fit(y_true.reshape(-1,1))
    y_true = ohe.transform(y_true.reshape(-1,1)).toarray()
    y_pred = ohe.transform(y_pred.reshape(-1,1)).toarray()

    y_true, y_predict = get_samples_bin(y_true, y_pred, 100)

    print(f"[MY] jaccard:   {jaccard_score(y_true, y_predict, 100)}")
    print(f"[SK] jaccard:   {sk_jacc(y_true, y_predict, average=None)}")
    print()
    
    print(f"[MY] accuracy:  {accuracy_score(y_true, y_predict, 100)}")
    print(f"[SK] accuracy:  {sk_acc(y_true, y_predict)}")
    print()
    
    print(f"[MY] precision: {precision_score(y_true, y_predict, 100)}")
    print(f"[SK] precision: {prec(y_true, y_predict, average=None)}")
    print()
    
    print(f"[MY] recall:    {recall_score(y_true, y_predict, 100)}")
    print(f"[SK] recall:    {recall(y_true, y_predict, average=None)}")
    print()
    
    print(f"[MY] f1_score:  {f1_score(y_true, y_predict, 100)}")
    print(f"[SK] f1_score:  {f1(y_true, y_predict, average=None)}")
    print()
    
    print(f"[MY] lift:      {lift_score(y_true, y_predict, 100)}")
    confusion_matrix(y_true, y_predict, 100)