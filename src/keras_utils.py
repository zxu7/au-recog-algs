import keras.backend as K
from keras.metrics import binary_accuracy

epsilon = 1e-7


def accuracy(y_true, y_pred):
    return K.mean(K.mean(K.equal(y_true, K.round(y_pred)), axis=-1))


def f_beta(y_true, y_pred, beta=1):
    tp = K.sum(K.round(y_pred * y_true), axis=-1)
    fp = K.sum(K.round(y_pred * K.abs(y_true - 1)), axis=-1)  # gtneg predpos
    fn = K.sum(y_true, axis=-1) - K.sum(K.round(y_pred * y_true), axis=-1)  # gtpos predneg <=> gtpos - gtpos predpos
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    # return K.mean((1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall))
    return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + epsilon)


def combo(y_true, y_pred, beta=1):
    acc = K.mean(K.mean(K.equal(y_true, K.round(y_pred)), axis=-1))
    tp = K.sum(K.round(y_pred * y_true), axis=-1)
    fp = K.sum(K.round(y_pred * K.abs(y_true - 1)), axis=-1)  # gtneg predpos
    fn = K.sum(y_true, axis=-1) - K.sum(K.round(y_pred * y_true), axis=-1)  # gtpos predneg <=> gtpos - gtpos predpos
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    # f1 = K.mean((1 + K.square(beta)) * (precision*recall) / (K.square(beta)*precision + recall))
    f1 = K.mean((1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + epsilon))
    return 0.5 * (acc + f1)
