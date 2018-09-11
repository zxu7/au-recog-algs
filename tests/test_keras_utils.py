import os
import sys
import numpy as np
import tensorflow as tf
import keras.backend as K
from sklearn.metrics import precision_score, recall_score

sys.path.append(os.path.abspath('..'))
from src.keras_utils import accuracy, f_beta


def test_accuracy():
    ph_y_true = K.placeholder((5, 5), dtype=K.tf.float32)
    ph_y_pred = K.placeholder((5, 5), dtype=K.tf.float32)

    y_true = np.eye(5)
    y_pred = np.zeros((5, 5))
    result = accuracy(ph_y_true, ph_y_pred)
    gt_result = 0.8

    with tf.Session() as sess:
        result = sess.run(result, feed_dict={
            ph_y_pred: y_pred,
            ph_y_true: y_true
        })

    print(result)
    print(gt_result)
    # assert result == gt_result, \
    #     "result doesn't match! prediction: {}, ground truth: {}".format(result, gt_result)


def test_f_beta(beta=1):
    ph_y_true = K.placeholder((3, 5), dtype=K.tf.float32)
    ph_y_pred = K.placeholder((3, 5), dtype=K.tf.float32)

    y_true = np.eye(5)[:3]
    y_pred = np.array([
        [1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    result = f_beta(ph_y_true, ph_y_pred, beta=-1)
    gt_precision = np.array([(lambda x, y: precision_score(x, y))(x, y) for (x, y) in zip(y_true, y_pred)])
    gt_recall = np.array([(lambda x, y: recall_score(x, y))(x, y) for (x, y) in zip(y_true, y_pred)])
    gt_result = (1 + beta ** 2) * (gt_precision * gt_recall) / (beta ** 2 * gt_precision + gt_recall)

    with tf.Session() as sess:
        result = sess.run(result, feed_dict={
            ph_y_pred: y_pred,
            ph_y_true: y_true
        })

    print(result)
    print(gt_precision)
    print(gt_recall)
    print(gt_result)
    # assert result == gt_result, \
    #     "result doesn't match! prediction: {}, ground truth: {}".format(result, gt_result)


if __name__ == '__main__':
    test_accuracy()
    test_f_beta()
