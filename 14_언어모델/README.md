# 자연어 처리와 언어모델

단어 벡터에서 시작해 RNN, seq2seq, 어텐션, Transformer, GPT까지 가는 강의 노트, 슬라이드와 코드.

- `main.pdf` : 강의 노트 (`main.tex`를 XeLaTeX로 컴파일한 것)
  - 단어를 벡터로
    1. 단어의 의미를 어떻게 표현할까
    2. 세어서 만드는 단어 벡터
    3. 맞히면서 배우는 단어 벡터: word2vec
    4. 큰 어휘를 위한 word2vec
  - 문장을 다루는 신경망
    5. 언어모델
    6. RNN
    7. LSTM: 오래 기억하는 RNN
    8. 문장 생성과 seq2seq
  - 어텐션과 Transformer
    9. 어텐션
    10. 어텐션만으로: Transformer
    11. GPT: Transformer로 만든 언어모델
- `slides_lm1.pdf`, `slides_lm2.pdf`, `slides_lm3.pdf` : 강의 슬라이드. 노트의 1–4절, 5–8절, 9–11절 (`slides_lm1.tex` 등, 공통 설정은 `slides_preamble.tex`)
- `code/` : 노트의 모든 코드. `code/` 폴더에서 실행하면 그림은 `figures/`에 저장된다.
- `figures/` : 노트와 슬라이드에 들어간 그래프 (코드가 만든 것)
- `tikz/` : 노트와 슬라이드에 들어간 그림 (TikZ, `main.tex`와 슬라이드가 읽어 들인다)
- `data/` : 노트의 숫자가 나온 실행 기록 (`*_log.txt`)과 `rnnlm_results.json`

## 코드

| 파일 | 내용 |
|---|---|
| `text_utils.py` | 공통 함수: 말뭉치 전처리, PTB 내려받기, 동시발생 행렬, PPMI, 코사인 유사도 |
| `count_based.py` | 세어서 만드는 단어 벡터: 동시발생 행렬, PPMI, SVD (작은 예문과 PTB) |
| `cbow_simple.py` | 문장 하나로 학습하는 CBOW 모델 |
| `cbow_ptb.py` | Embedding과 네거티브 샘플링을 쓴 CBOW, PTB (GPU 권장) |
| `rnnlm_ptb.py` | RNN과 LSTM 언어모델, 문장 생성, PTB (GPU 권장) |
| `seq2seq_addition.py` | 덧셈을 배우는 seq2seq: 기본, 입력 뒤집기, 엿보기 (GPU 권장) |
| `attention_date.py` | 어텐션을 넣은 seq2seq로 날짜 형식 바꾸기, 어텐션 가중치 그림 (GPU 권장) |
| `lm_tiny.py` | 손으로 따라갈 수 있는 작은 예: self-attention, 위치 벡터 (사인파 포함), temperature, top-k와 top-p, LayerNorm, √d로 나누는 이유 |
| `gpt_ptb.py` | 작은 GPT (Decoder만 쓰는 Transformer) 언어모델, PTB (GPU 권장) |
| `bpe_tokens.py` | BPE로 토큰 만들기: 작은 예와 PTB |
| `figures_extra.py` | 설명용 그래프: 차원 축소, 오답 샘플링 분포, RNN의 기울기, 퍼플렉서티, 어텐션 가중치 |

`gpt_ptb.py`는 `rnnlm_ptb.py`가 저장한 `data/rnnlm_results.json`을 읽어 세 모델을 한 그래프에 그린다.

## 데이터

큰 실험은 말뭉치 PTB (Penn Treebank)를 쓴다. 저장소에는 들어 있지 않고,
PTB를 쓰는 코드를 처음 실행할 때 `data/` 폴더로 내려받는다 (`ptb.train.txt`, `ptb.valid.txt`, `ptb.test.txt`, 합쳐서 약 6 MB).
덧셈 문제와 날짜 데이터는 코드가 직접 만든다.

## 참고

노트의 구성은 『밑바닥부터 시작하는 딥러닝 2』 (사이토 고키 지음, 한빛미디어)의 2–8장의 순서를 따랐다.
코드는 PyTorch로 새로 썼고, 그림은 새로 그렸다.
