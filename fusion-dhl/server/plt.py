from matplotlib import pyplot as plt
from scipy.optimize import least_squares
import pandas as pd
import numpy as np
from enum import Enum

pos = np.array([
    [0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0],
    [10,0],[11,0],[12,0],[13,0],[14,0],[15,0],[16,0],[17,0],[18,0],[19,0],
    [20,0],[21,0],[22,0],[23,0],[24,0],[25,0]
])

df = pd.read_csv('server/pos_csv/1732617492733_pos.csv')

class Function:
    def __init__(self):
        self.c_pos = None
        self.c_radius = 1
    
    def set_pos(self, pos):
        self.c_pos = pos
    
    def residuals(self, pos):
        pos = np.array(pos).reshape(-1, 2)
        
        dist = np.linalg.norm(pos - self.c_pos, axis=1)
        return np.maximum(dist - self.c_radius / 2, 0)
    
    def __call__(self, pos):
        return self.residuals(pos)

fun = Function()
fun.set_pos(pos)
initial_params = np.zeros(26*2).tolist()
for i in range(1000):
    result = least_squares(fun, initial_params)
    initial_params = result.x
new_pos = np.array(initial_params).reshape(-1, 2)
# print(initial_params.tolist())
fig,ax = plt.subplots(1,1,figsize = (10,10))
ax.scatter(pos[:,0],pos[:,1],s=5,c='red',label='Original Trajectory')

ax.scatter(new_pos[:,0],new_pos[:,1],s=5,c='blue',label='Ang Trajectory')
# ax.plot(df['x'],fitted_y,c='green',label='Fitted Trajectory')
plt.show()