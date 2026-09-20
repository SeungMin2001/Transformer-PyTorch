# Transformer (PyTorch)

[🇰🇷 한국어](README.md) | [🇺🇸 English](README.en.md)

> **Attention Is All You Need**를 바탕으로 Transformer를 처음부터 구현하며 학습하기 위한 PyTorch 프로젝트입니다.

<p align="left">
  <img src="assets/transformer_architecture.jpg" alt="Transformer 아키텍처" width="350">
</p>
<p align="left">
  <em>그림. Transformer 아키텍처</em>
</p>

## 개요

이 저장소는 원래 Transformer 구조를 학습 중심으로 구현한다. README의 개념 노트와 src/의 재사용 가능한 PyTorch 모듈을 함께 제공한다.

- sqrt(d_model)을 적용한 토큰 임베딩 스케일링
- 사인·코사인 위치 인코딩
- Scaled Dot-Product Attention과 Multi-Head Attention
- 인코더 self-attention, 디코더 masked self-attention, 인코더-디코더 attention
- residual connection, dropout, LayerNorm
- teacher forcing을 사용하는 sequence-to-sequence 학습

## 논문

> **Attention Is All You Need**  
> Vaswani et al., *NeurIPS 2017*

[arXiv: 1706.03762](https://arxiv.org/abs/1706.03762)

## 실행 가능한 구현

- src/transformer_pytorch/model.py: 토큰 임베딩, 위치 인코딩, multi-head attention, FFN, encoder·decoder layer, 전체 encoder-decoder Transformer
- examples/toy_forward.py: (batch, target_length, vocab_size) 출력 형태를 확인하는 작은 forward-pass 예제
- tests/test_transformer.py: 출력 형태, 위치 인코딩, causal mask 테스트

~~~bash
pip install -e ".[dev]"
PYTHONPATH=src python examples/toy_forward.py
PYTHONPATH=src pytest -q
~~~

구현의 핵심 흐름은 다음과 같다.

~~~text
임베딩 × sqrt(d_model) → 위치 정보 추가 → QKᵀ / sqrt(d_k) attention
→ decoder causal mask → attention·FFN마다 residual Add & Norm
~~~

## 학습 노트

### 1. 데이터셋과 토크나이저

Multi30k와 Hugging Face AutoTokenizer를 사용해 입력을 토큰화하고, 배치 단위로 padding과 tensor 변환을 처리한다.

### 2. 임베딩과 위치 인코딩

토큰 임베딩은 sqrt(d_model)로 스케일링한다. 사인·코사인 위치 인코딩을 더해 순서 정보를 제공하며, 이 값은 학습 가능한 파라미터가 아니다.

### 3. Attention

Scaled dot-product attention은 QKᵀ / sqrt(d_k)를 softmax에 전달한다. Multi-head attention은 여러 표현 부분 공간에서 이 연산을 병렬 수행한 뒤 결합한다. 디코더 self-attention에는 미래 토큰을 가리는 causal mask를 적용한다.

### 4. Feed-Forward Network와 모델링

각 Transformer 블록은 attention 뒤와 feed-forward network 뒤에 residual connection, dropout, LayerNorm을 적용한다. 인코더는 입력 표현을 만들고, 디코더는 이전 출력과 인코더 표현을 함께 사용해 다음 토큰 분포를 생성한다.

### 5. 학습과 평가

teacher forcing으로 타깃 시퀀스를 한 칸 이동시켜 학습하며, padding 토큰을 제외한 cross-entropy loss로 최적화한다. 학습·검증 손실 곡선과 예시 생성 결과로 모델을 확인한다.

상세한 코드 조각과 원문 학습 노트는 [English README](README.en.md)에서 볼 수 있다.

