# 딥러닝 개발자를 위한 하드웨어 (읽기 자료)

- `main.pdf` : 읽기 자료 (`main.tex`를 XeLaTeX로 컴파일한 것)
  1. 컴퓨터 한 대를 열어 보면
  2. 메인보드와 PCIe
  3. 메모리
  4. GPU의 안
  5. 숫자 형식과 정밀도
  6. 성능을 재는 단위
  7. GPU의 세대와 종류
  8. 소프트웨어 스택
  9. 개발 환경
  10. 사례: 연구용 GPU 워크스테이션
  11. 용어 정리
- `code/` : 측정과 그래프 코드
- `figures/` : 노트에 들어간 그래프와 사진. `figures/photos/`의 사진은 모두 Wikimedia Commons에서 받은 것이고, 저작자와 라이선스는 `figures/photos/CREDITS.md`와 노트 끝의 사진 출처 표에 있다.
- `results/` : 강의자의 노트북 (RTX 3050 4GB Laptop GPU)에서 `device_info.py`와 `gpu_bench.py`로 잰 값

## 코드

| 파일 | 내용 |
|---|---|
| `device_info.py` | GPU, SM 수, VRAM, compute capability, CUDA 버전 출력 |
| `gpu_bench.py` | 행렬곱 속도와 정확도, VRAM과 PCIe의 대역폭 측정 |
| `hardware_charts.py` | 대역폭 비교, GPU별 성능 그래프 (사양 값) |
| `photo_credits.py` | `figures/photos/CREDITS.md`로 노트 끝의 사진 출처 표 (`photo_credits.tex`)를 만든다 |

실행: `code` 폴더에서 `python gpu_bench.py` 처럼 실행한다. `device_info.py`와 `gpu_bench.py`는 PyTorch가 필요하고, `gpu_bench.py`는 NVIDIA GPU도 필요하다. `hardware_charts.py`는 matplotlib만 있으면 된다.
