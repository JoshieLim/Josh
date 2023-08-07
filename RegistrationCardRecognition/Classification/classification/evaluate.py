import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, \
    auc, confusion_matrix


def evaluate_model(plot=False):
    '''
    Evaluate the performance of the trained model using test dataset

    Args:
        plot: Plot ROC and PR auc
    Returns:
        None
    '''
    ROOT_DIR = Path().resolve().parent.parent
    TESTDIR = os.path.join(ROOT_DIR, 'data', 'registrationcard',
                           'test')
    size = 150
    batch = 32

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        TESTDIR, target_size=(size, size),
        class_mode='binary', color_mode='rgb', batch_size=batch,
        shuffle=False, classes=['img', 'NonRC'])

    model_path = os.path.join(ROOT_DIR, 'data', 'classifier_data', 'model',
                              'classifier_model.h5')
    model = load_model(model_path)

    evaluation = model.evaluate(test_generator)

    preds = model.predict(test_generator, steps=None)
    y_pred = np.rint(preds)
    y_true = test_generator.classes.ravel()

    # result_df = pd.DataFrame({'pred_y': y_pred.ravel(),
    #                           'true_y': test_generator.classes.ravel(),
    #                           'prob_y': preds.ravel()})

    roc = roc_auc_score(y_true, preds)
    fpr, tpr, _ = roc_curve(y_true, preds)
    precision, recall, thresholds = precision_recall_curve(y_true, preds)
    pr_auc = auc(recall, precision)

    print('##################### Model Evaluation ########################')
    print(f'Model accuracy: {evaluation[1]:<.2f}')
    print(f'Model ROC AUC: {roc:<.2f}')
    print(f'Model PR AUC: {pr_auc:<.2f}')
    print('Confusion matrix')
    print(confusion_matrix(y_true, y_pred))

    if plot:
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label='ROC-AUC: {:<.2f}'.format(roc))
        plt.plot([0, 1], [0, 1], linestyle='--', label='random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, marker='.',
                 label='PR-AUC: {:<.2f}'.format(pr_auc))
        plt.plot([0, 1], [0.5, 0.5], linestyle='--', label='random')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        plt.show()


if __name__ == "__main__":
    start = time.time()
    evaluate_model(plot=False)
    print('Evaluation took : {:.2f} seconds'.format(time.time() - start))
