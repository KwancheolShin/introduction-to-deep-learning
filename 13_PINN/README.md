# 물리 정보 신경망 (PINN)

- `main.pdf` : 강의 노트 (`main.tex`를 XeLaTeX로 컴파일한 것)
  1. PINN이란 무엇인가
  2. 신경망은 입력의 함수이다
  3. 계산 그래프로 끝까지 따라가기
  4. PyTorch에서 입력으로 미분하기
  5. 첫 번째 PINN: 1차원 Poisson 방정식
  6. 시간이 있는 문제: 열 방정식
  7. 역문제: 측정값에서 계수 찾기
  8. 2차원 타원형 방정식
  9. 역문제 2: 전기 임피던스 단층촬영 (EIT)
  10. PINN의 한계와 요령
- `slides_pinn.pdf` : 강의 슬라이드 (`slides_pinn.tex`)
- `code/` : 노트의 모든 코드. `code/` 폴더에서 실행하면 그림은 `figures/`에 저장된다.
- `figures/` : 노트와 슬라이드에 들어간 그림

## 코드

| 파일 | 내용 |
|---|---|
| `autograd_tiny.py` | 은닉 뉴런 1개짜리 신경망으로 $u$, $u_x$, $u_{xx}$, loss, gradient를 손계산과 비교 |
| `tanh_vs_relu.py` | tanh와 ReLU 신경망의 $u'$, $u''$ |
| `data_vs_physics.py` | 데이터만, 방정식만, 둘 다 쓴 경우의 비교 |
| `pinn_poisson_1d_simple.py` | 첫 PINN: $-u''=f$, 함수 없이 핵심만 |
| `pinn_poisson_1d.py` | 같은 문제에 기록, 그림, L-BFGS, 유한차분법 비교를 더한 것 |
| `pinn_heat_1d.py` | 열 방정식 $u_t=\kappa u_{xx}$ (GPU 권장) |
| `pinn_inverse_heat.py` | 역문제: 측정값에서 $\kappa$ 찾기 (GPU 권장) |
| `fem_disk.py` | 원판에서의 유한요소법: 기준 해와 EIT 데이터 |
| `pinn_elliptic_2d.py` | 2차원 타원형 방정식 $\nabla\cdot(\sigma\nabla u)=0$ |
| `eit_boundary_data.py` | EIT 데이터 한 쌍 $(g_1,f_1)$을 중심각과 원판 둘레에 그리기 |
| `pinn_eit.py` | EIT: 경계의 전류--전압 쌍에서 $\sigma$ 찾기 (GPU에서 수 시간) |
| `pinn_frequency.py` | 높은 주파수의 해와 collocation point의 수 |
