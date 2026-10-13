# 합성곱 신경망 (CNN)

밑바닥부터 시작하는 딥러닝 7장에 대응하는 강의 자료입니다.

- `main.pdf` : 강의 노트 (`main.tex`를 XeLaTeX로 컴파일한 것)
  1. 전체 구조
  2. 합성곱 계층: 합성곱 연산, 패딩, 스트라이드, 3차원 데이터, 블록으로 생각하기, 배치 처리
  3. 풀링 계층
  4. 합성곱 / 풀링 계층 구현하기: im2col, Convolution, Pooling, Flatten과 역전파
  5. CNN 구현하기: SimpleConvNet
  6. CNN 시각화하기
  7. 대표적인 CNN: LeNet, AlexNet
  8. PyTorch로 구현하기
- `code/` : 노트의 모든 코드. 그림은 `figures/`에 저장된다.
- `figures/` : 노트에 들어간 그림

## 코드

| 파일 | 내용 |
|---|---|
| `conv_layers.py` | `im2col`, `col2im`, `Convolution`, `Pooling`, `Flatten`, `SimpleConvNet` |
| `layers.py`, `optimizers.py`, `mnist_data.py` | 학습 관련 기술 주차의 모듈 (그대로 복사) |
| `conv_basics.py` | 노트의 예제 (합성곱, 패딩, 스트라이드, 풀링), im2col 확인과 속도 비교, gradient 확인 |
| `mnist_cnn.py` | SimpleConvNet으로 MNIST 학습 (MLP와 비교), 필터와 특징 맵 시각화. 약 13분 걸린다 |
| `mnist_cnn_torch.py` | 같은 SimpleConvNet을 PyTorch로 |

실행: `code` 폴더에서 `python conv_basics.py` 처럼 실행한다 (numpy, matplotlib, 그리고 MNIST를 내려받을 scikit-learn 필요. `mnist_cnn_torch.py`는 PyTorch 필요).
