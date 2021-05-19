
from seqeval import metrics
import torch

def run():
    # y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    # y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]

    y_true = [['B-PER', 'I-PER', 'O'], ['B-PER', 'I-PER', 'E-PER', 'O']]
    y_pred = [['B-PER', 'I-PER', 'O'], ['B-LOC', 'I-LOC', 'E-LOC', 'O']]

    print(metrics.precision_score(y_true, y_pred))
    print(metrics.recall_score(y_true, y_pred))
    print(metrics.f1_score(y_true, y_pred))
    print(metrics.performance_measure(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred))


if __name__ == '__main__':
    run()
