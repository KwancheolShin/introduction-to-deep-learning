# 다중분류 (MNIST)

MNIST 손글씨 숫자 분류의 실습 코드와 코드 읽기 슬라이드입니다.

## 코드

| 파일 | 내용 |
|---|---|
| `mnist_lab.py` | NumPy로 만든 은닉층 1개의 신경망 (계층 클래스, gradient 확인, 미니배치 학습) |
| `mnist_mlp.py` | 같은 계층으로 은닉층을 여러 개 쌓은 `MultiLayerNet`, 훈련 / 검증 / 시험 분할, 혼동행렬 |
| `mnist_mlp_simple.py` | 실험 없이 network 하나 (`[784, 100, 10]`)를 `net.fit(...)` 메서드로 학습 (Keras의 `model.fit`처럼 학습 반복을 클래스 안에 둠). `mnist_torch.py`와 대조 |
| `mnist_torch.py` | 같은 MNIST 분류를 PyTorch로 |
| `loop_vs_matmul.py` | 샘플마다 for문으로 계산할 때와 행렬곱 한 번으로 계산할 때의 속도 비교 |
| `mnist_intro_figures.py` | MNIST 예시 이미지, 이미지를 784차원 행벡터로 펼치는 그림 |
| `mlp_demo.py` | `mnist_mlp.py` 슬라이드의 작은 예제 (one-hot, shape, gradient 확인, softmax) |
| `torch_basics.py` | PyTorch 입문 슬라이드의 작은 예제 (텐서, autograd, nn, optim, utils.data, train / eval 모드, 데코레이터) |

## 슬라이드

- `slides_numpy_mnist_mlp.pdf` : `mnist_mlp.py` 코드 읽기 (`.tex`)
- `slides_pytorch_mnist_torch.pdf` : PyTorch 입문과 `mnist_torch.py` 코드 읽기 (`.tex`)
- 두 슬라이드의 그림은 `slides_figures/`에 있다.

실행: 이 폴더에서 `python mnist_mlp.py` 처럼 실행한다 (numpy, matplotlib, 그리고 MNIST를 내려받을 scikit-learn 필요. PyTorch 코드는 PyTorch 필요).
