# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:30:30 2025

@author: kcshi
"""

import numpy as np
import matplotlib.pyplot as plt

N = 5
x = np.array([-2,-1,0,1,2])
t = np.array([0,0,1,1,1]) #targets

plt.plot(x,t,'ro')
plt.xlabel('x')
plt.ylabel('t')
plt.title("data point")

plt.show()

phi = lambda u: 1/(1+np.exp(-u))
#dphi = lambda u:np.exp(-u)/(1+np.exp(-u))**2
dphi = lambda u:phi(u)*(1-phi(u))

# 2. 초기 파라미터 설정
w = 0.0  # 기울기
b = 0.0  # 절편

def compute_cost(x, t, w, b):
    # x: 입력변수
    # y: 출력변수
    # w: 웨이트
    # b: 바이어스
    u = w * x + b
    cost   = 1/2*np.mean((phi(u) -t)**2)
    return cost

# 하이퍼파라미터
learning_rate = 0.001
epochs = 30000
n = len(x)  # 데이터 개수

# 4. 경사하강법 학습 루프
cost_history = []

for i in range(epochs):
    u = w * x + b
    t_pred = phi(u)
    error = t_pred - t
    dphi_u = dphi(u)
    # 기울기 계산
    dw = (2 / n) * np.sum(error*dphi_u*x)
    db = (2 / n) * np.sum(error*dphi_u)

    # 기울기 계산
    #dw = (2 / n) * np.sum(error*t_pred*(1-t_pred)*x)
    #db = (2 / n) * np.sum(error*t_pred*(1-t_pred))


    # 파라미터 업데이트
    w -=  learning_rate * dw
    b -=  learning_rate * db

    # 비용 기록
    cost = compute_cost(x, t, w, b)
    cost_history.append(cost)

    # 중간 출력
    if i % 100 == 0:
        print(f"Epoch={i}: Cost={cost:.4f}, w={w:.4f}, b={b:.4f}")

# 5. 최종 결과 출력
print("\n최종 결과:")
print(f"w = {w:.4f}")
print(f"b = {b:.4f}")


# 6. 회귀선 시각화
xx = np.linspace(-2.5,2.5)
t_fit = phi(w * xx + b)
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.plot(x, t, 'ro')
plt.plot(xx, t_fit, 'k', label='regression')
plt.xlabel('x')
plt.ylabel('t')
plt.title('nonlinear regression')
plt.legend()
plt.grid(True)

# 7. 비용 함수 감소 시각화
plt.subplot(1, 2, 2)
plt.plot(cost_history, label='Cost (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost History')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


