
import numpy as np
import matplotlib.pyplot as plt


ff = lambda x, y: 3*(x-6)**2 + (y-5)**2
fx = lambda x, y: 6*(x-6)
fy = lambda x, y: 2*(y-5)

ox, oy = 1, 1 #초깃값 설정
epochs = 100
lr = 0.05 #0.25 0.3

xp, yp = np.zeros(epochs+1), np.zeros(epochs+1)

xp[0], yp[0] = ox, oy

for i in range(epochs):
    gradx, grady = fx(ox,oy), fy(ox,oy)   # Compute gradient
    nx, ny = ox - lr*gradx, oy - lr*grady # Gradient Descent
    ox, oy = nx, ny                 # Update Iteration
    xp[i+1], yp[i+1] = ox, oy

l_bdr, r_bdr = 0, 10
x = np.linspace(l_bdr,r_bdr,100)
y = np.linspace(l_bdr,r_bdr,100)

xx, yy = np.meshgrid(x,y) # 격자점 생성
zz = ff(xx,yy)

# 등고선 그림
fig1 = plt.figure()
ax1  = plt.axes()
ax1.contour(xx, yy, zz, 20, cmap='jet')
ax1.set_aspect('equal')
ax1.plot(6, 5, 'rs')
ax1.plot(xp, yp, 'ko-')

# 3차원 플롯
fig2 = plt.figure()
ax2  = plt.axes(projection='3d')
ax2.plot_surface(xx, yy, zz, cmap='jet')
ax2.scatter(6, 5, ff(6, 5), color='r', s=50)
ax2.plot(xp, yp, ff(xp, yp), 'k.')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x, y)')

plt.show()