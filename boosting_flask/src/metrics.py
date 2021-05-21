import numpy as np
from sklearn.metrics import f1_score


def f1_multiclass(y_hat, train_data):
    y_true = train_data.get_label( )
    y_hat = y_hat.reshape(-1, -1).T
    y_hat = np.argmax(y_hat, axis=1)
    return 'f1', f1_score(y_true, y_hat, average='macro'), True


def f1_binary(y_hat, train_data):
    y_true = train_data.get_label( )
    y_hat = np.where(y_hat < 0.5, 0, 1)
    return 'f1', f1_score(y_true, y_hat, average='binary'), True
