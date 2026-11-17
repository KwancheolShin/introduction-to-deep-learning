# 딥러닝 (Deep Learning)

밑바닥부터 시작하는 딥러닝 8장에 대응하는 강의 자료입니다.

- `main.pdf` : 강의 노트 (`main.tex`를 XeLaTeX로 컴파일한 것)
  1. 더 깊게: 더 깊은 신경망 (DeepConvNet), 정확도를 더 높이려면 (데이터 확장), 층을 깊게 하는 이유
  2. 딥러닝의 초기 역사: 이미지넷, VGG, GoogLeNet, ResNet
  3. 더 빠르게: 시간은 어디에 쓰이는가, GPU, 분산 학습, 연산 정밀도
  4. 딥러닝의 활용: 사물 검출, 분할, 사진 캡션 생성
  5. 딥러닝의 미래: 이미지 스타일 변환, 이미지 생성, 자율 주행, 강화학습
- `code/` : 노트의 모든 코드. 그림은 `figures/`에 저장된다.
- `figures/` : 노트에 들어간 그림. 논문에서 가져온 그림은 노트의 그림 설명에 출처가 있다.
- `results/` : 코드를 실행한 기록. `deep_convnet_params.npz`는 `mnist_deep_torch.py`로 학습한 매개변수이고, `speed_precision.py`가 이것을 읽는다.
- `연습문제_정답/` : 노트 끝 연습문제의 풀이

## 코드

| 파일 | 내용 |
|---|---|
| `deep_convnet.py` | NumPy로 만든 DeepConvNet. 실행하면 기울기 확인을 한다 |
| `mnist_deep_torch.py` | 같은 network를 PyTorch로 학습 (데이터 확장 선택 가능) |
| `augmentation.py` | 데이터 확장 (작은 회전과 이동) |
| `resnet_skip.py` | skip connection이 있을 때와 없을 때의 학습 비교 |
| `speed_precision.py` | 계층별 계산 시간, NumPy와 PyTorch의 속도, 16비트 정확도 |
| `ilsvrc_chart.py` | ImageNet 대회 오류율 그래프 |
| `layers.py`, `conv_layers.py`, `optimizers.py`, `mnist_data.py` | 앞 단원에서 만든 모듈 |

실행: `code` 폴더에서 `python deep_convnet.py` 처럼 실행한다 (numpy, matplotlib, 그리고 MNIST를 내려받을 scikit-learn 필요. `mnist_deep_torch.py`, `augmentation.py`, `resnet_skip.py`, `speed_precision.py`는 PyTorch 필요).
