# 학습 관련 기술 (Techniques for Training)

밑바닥부터 시작하는 딥러닝 6장에 대응하는 강의 자료입니다.

- `main.pdf` : 강의 노트 (`main.tex`를 XeLaTeX로 컴파일한 것)
  1. 매개변수 갱신: SGD, Momentum, AdaGrad, RMSProp, Adam
  2. 가중치의 초깃값: Xavier, He
  3. 배치 정규화
  4. 바른 학습을 위해: 오버피팅, 가중치 감소, AdamW, 드롭아웃
  5. 적절한 하이퍼파라미터 값 찾기
  - 부록: 실제 딥러닝의 optimizer, SGD의 수렴 조건, L-BFGS와 2차 미분을 쓰는 방법
- `slides/` : 강의 슬라이드
- `code/` : 노트의 모든 코드. 그림은 `figures/`에 저장된다.
- `figures/` : 노트에 들어간 그림

## 코드

모듈 (다른 코드가 불러 쓰는 파일)

| 파일 | 내용 |
|---|---|
| `optimizers.py` | SGD, Momentum, AdaGrad, RMSProp, Adam, AdamW (`update(params, grads)`) |
| `layers.py` | Affine, ReLU, Sigmoid, Softmax-loss, BatchNormalization, Dropout, MultiLayerNet |
| `mnist_data.py` | MNIST 불러오기, 훈련 / 검증 / 시험 분할 |

실험 코드

| 파일 | 노트 | 내용 |
|---|---|---|
| `optimizers_2d.py` | 1절 | 2변수 함수 f(x, y) = x²/20 + y²에서 네 기법의 경로 |
| `mnist_optimizer_compare.py` | 1.9절 | MNIST에서 optimizer 비교 |
| `weight_init_activation.py` | 2절 | 초깃값에 따른 은닉층 활성화값 분포 |
| `mnist_init_compare.py` | 2.5절 | MNIST에서 초깃값 비교 |
| `mnist_batchnorm.py` | 3절 | 배치 정규화의 gradient 확인과 효과 |
| `mnist_overfit.py` | 4절 | 오버피팅, 가중치 감소, 드롭아웃, AdamW |
| `hyperparameter_search.py` | 5절 | 학습률과 가중치 감소의 무작위 탐색 |
| `sgd_noise_2d.py`, `lbfgs_2d.py` | 부록 | SGD의 noise floor, L-BFGS (PyTorch) |

실행: `code` 폴더에서 `python mnist_batchnorm.py` 처럼 실행한다 (numpy, matplotlib, 그리고 MNIST를 내려받을 scikit-learn 필요. `lbfgs_2d.py`는 PyTorch 필요).
