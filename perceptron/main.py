import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from percetron import Perceptron
from plotter import Plotter


df = pd.read_csv(
    'resources/apendice.data', header=None)

df.tail()

y = df.iloc[0:30, 4].values
y = np.where(y == 'CLASSEP1', -1, 1)
X = df.iloc[0:30, [0, 1, 2]].values

ppn = Perceptron(taxa_aprendizado=0.01, epocas=30)
ppn.treinar(X, y)

# Plotter.plot_decision_regions(X, y, ppn)

# Plotter.plot_bidimensional_features(X, 0, 1)
# Plotter.plot_bidimensional_features(X, 0, 2)
# Plotter.plot_bidimensional_features(X, 1, 2)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero de classificações erradas')
plt.show()
