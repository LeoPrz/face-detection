import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_rects(img, rects, ax=None, **kwargs):
    kwargs.setdefault('edgecolor', 'r')
    kwargs.setdefault('linewidth', 1)

    if ax is None:
        fig, ax = plt.subplots(1)
    ax.imshow(img)
    for r in rects:
        rect = patches.Rectangle((r.x, r.y), r.w, r.h, facecolor='none', **kwargs)
        ax.add_patch(rect)

    plt.show()


def plot_images(imgs, size=5):
    for i in range(size ** 2):
        plt.subplot(size, size, i + 1)
        plt.axis('off')
        plt.imshow(imgs[i])

    plt.subplots_adjust(.01, .01, 1, 1, .01, .01)
    plt.show()


def plot_precision_recall_f1(prec, recall, f1, thresholds):
    plt.subplot(1, 2, 1)
    plt.plot(recall, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, f1[:-1])
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.show()
