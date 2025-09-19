import torch
from torch import Tensor

from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def show_f1_results(
    model_name: str,
    f1_score_each: Tensor,
    class_ls: list,
    show_details=False
):
    f1_score_avg = f1_score_each.mean(dim=0, keepdim=True)
    f1_score_min = f1_score_each.min().unsqueeze(0)

    if show_details:
        pd.set_option('display.float_format', '{:.2f}'.format)
        
        f1_score_all = torch.cat([f1_score_avg, f1_score_min, f1_score_each]).numpy()

        df = pd.DataFrame(
            data={
                model_name: ['Avg', 'Min'] + class_ls,
                'F1-score': f1_score_all
            },
            index=['avg', 'min'] + list(range(len(class_ls)))
        )
        print(df)
    
    else:
        pd.set_option('display.float_format', '{:.3f}'.format)
        
        f1_score_all = torch.cat([f1_score_avg, f1_score_min]).unsqueeze(0).numpy()
        df = pd.DataFrame(
            data=f1_score_all,
            columns=['Avg', 'Min'],
        )
        df[model_name] = 'F1-score'
        df.set_index(model_name, drop=True, inplace=True)
        print(df)


def set_one_ax_confusion_matrix(
    fig: Figure,
    ax: Axes,
    matrix: np.ndarray,
    normalized=True,
    font_size=12
):
    classes = np.arange(len(matrix))
    ax_img = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(ax_img, ax=ax)
    ax.set_xticks(classes, classes, fontsize=font_size)
    ax.set_yticks(classes, classes, fontsize=font_size)
    
    if len(matrix) < 18:  # 類別少的話可以顯示個別的數值了
        fmt = '.2f' if normalized else 'd'
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, format(matrix[i, j], fmt),
                    verticalalignment='center',
                    horizontalalignment="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=font_size
                )
    else:  # draw grids
        for i in classes[:-1]:
            mid_point = (classes[i] + classes[i + 1]) / 2
            ax.axvline(x=mid_point, color='black', linestyle='-')
            ax.axhline(y=mid_point, color='black', linestyle='-')

    ax.set_xlabel('Prediction', fontsize=font_size)
    ax.set_ylabel('Ground Truth', fontsize=font_size)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    need_pre_argmax: bool,
    model_name: str,
    font_size=12,
    save_name=None,
    save=True
):
    '''`save_name` is default = `model_name`'''
    if need_pre_argmax:
        matrix = confusion_matrix(
            np.argmax(y_true, axis=1),
            np.argmax(y_pred, axis=1)
        )
    else:
        matrix = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f'{model_name} Result On Testing Set')
    ax1, ax2 = fig.subplots(1, 2)
    ax1: Axes; ax2: Axes

    precision_m = matrix.astype(np.float32) / matrix.sum(axis=0)
    ax1.set_title('Confusion Matrix (Precision)')
    set_one_ax_confusion_matrix(fig, ax1, precision_m, normalized=True, font_size=font_size)
    
    recall_m = matrix.astype(np.float32) / matrix.sum(axis=1, keepdims=True)
    ax2.set_title('Confusion Matrix (Recall)')
    set_one_ax_confusion_matrix(fig, ax2, recall_m, normalized=True, font_size=font_size)

    if save_name is None:
        save_name = model_name
    if save:
        plt.savefig(f'{save_name}_confusion_matrix.jpg')
    else:
        plt.show()


if __name__ == '__main__':
    n_classes = 35

    y_true = np.eye(n_classes).repeat(4, axis=0)

    arr = np.arange(n_classes)
    np.random.shuffle(arr)
    y_pred = np.eye(n_classes)[arr].repeat(4, axis=0)

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        need_pre_argmax=True,
        model_name='Example',
        normalize_strategy='precision',
        font_size=6,
        save=False
    )
