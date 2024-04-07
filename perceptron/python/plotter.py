from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def plot_decision_regions(X, y, classifier, resolution=0.02):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(
            np.array([xx1.ravel(), xx2.ravel()]).T)

        Z = Z.reshape(xx1.shape)

        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                        c=cmap(idx), marker=markers[idx], label=cl)

    def plot_bidimensional_features(X, idx_feature1, idx_feature2):
        plt.scatter(X[:14, idx_feature1], X[:14, idx_feature2], color='red',
                    marker='o', label='Classe P1')
        plt.scatter(X[14:30, idx_feature1], X[14:30, idx_feature2], color='blue',
                    marker='x', label='Classe P2')
        plt.xlabel('Caracteristica ' + str(idx_feature1 + 1))
        plt.ylabel('Caracteristica ' + str(idx_feature2 + 1))
        plt.legend(loc='upper right')
        plt.show()
