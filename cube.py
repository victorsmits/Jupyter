import numpy as np
from matplotlib import pyplot as plt

x = [1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1]
y = [1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1]
z = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1]
# Certains sommets apparaissent plusieurs fois pour facilité l'affichage avec plot()

M = np.array([[1, 2, 3], [4, 5, 6]])
Xcube = np.array([x, y, z, [1]*16])
print(Xcube)

N = np.array([[1], [0], [0]])
print(M @ N)

# numpy supporte les notations indicielles de matlab
# ici on récupère les coordonnées classiques à partir
# des coordonnées homogènes en divisant par la quatrième composante
xi = Xcube[0, :]/Xcube[3, :]
yi = Xcube[1, :]/Xcube[3, :]
plt.plot(xi, yi)
plt.show() # voir figure 3

factor = 2.5
plt.figure(figsize=(4*factor, 3*factor)) # taille de l'image en pouces
plt.axis([0, 640, 480, 0])
plt.plot([50, 50, 100, 100, 50], [50, 100, 100, 50, 50], "-o") # "-o" pour traits et cercle
plt.show() # voir figure 4