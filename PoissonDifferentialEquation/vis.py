import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.read_csv("./res.txt", sep=',', header=None)

M = int(df.iloc[0,0])
N = int(df.iloc[0,1])

fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig)

x1 = np.linspace(0.0, 4.0, M + 1)
x2 = np.linspace(0.0, 3.0, N + 1)
X, Y = np.meshgrid(x1, x2)
Z = np.sqrt(4 + X * Y)
plt.title("the real value")
plt.xlabel('x')
plt.ylabel('y')
# ax.plot_surface(X, Y, Z, alpha = 0.9, rstride=1, cstride=1, cmap="rainbow")
ax.plot_surface(X, Y, Z, linewidth = 0, cmap=cm.coolwarm, antialiased=False)

plt.show()

res = []
df_points = df.iloc[1:]
for i in range(M+1):
    res.append(df_points.iloc[i * (N + 1) : (i + 1) * (N + 1), 1])

fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig)

x1 = np.linspace(0.0, 4.0, M + 1)
x2 = np.linspace(0.0, 3.0, N + 1)
X, Y = np.meshgrid(x1, x2)
Z = np.array(res)
plt.title("the simulated value")
plt.xlabel('x')
plt.ylabel('xy')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax.plot_surface(X, Y, Z, linewidth = 0, cmap=cm.coolwarm, antialiased=False)
plt.show()
