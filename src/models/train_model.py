from model import create_model
from ..data import DataLoader
from pathlib import Path
import numpy as np

import os
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


def shuffle(x_train, y_train):
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    return x_train[idx], y_train[idx]


def plot_loss_graph(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('./../../reports/figures/loss-curve.png')

    return 0


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('./../../reports/figures/confusion_matrx.png')

    return 0


if __name__ == '__main__':

    model = create_model()

    project_dir = Path(__file__).resolve().parents[2]
    print("Project dir {0}".format(project_dir))

    loader = DataLoader(os.path.join(project_dir, "data/external"))
    x_train, x_test, y_train, y_test = loader.load_data()

    x_train, y_train = shuffle(x_train, y_train)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    history = model.fit(x_train, y_train, batch_size=16, epochs=200, validation_data=(x_test, y_test))

    plot_confusion_matrix(history)
    y_pred_class = model.predict(x_test)
    y_pred_class[y_pred_class < 0.5] = 0
    y_pred_class[y_pred_class >= 0.5] = 1
    cm = confusion_matrix(y_test, y_pred_class)
    plot_confusion_matrix(cm,
                          normalize=False,
                          target_names=['normal', 'cataract'],
                          title="Confusion Matrix")

    model.save('final-700imgs.h5')
