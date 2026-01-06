# Transformer (PyTorch)
> **Attention Is All You Need** 논문을 기반으로 Transformer를 PyTorch로 처음부터 구현 <br>
> A PyTorch implementation of the Transformer built from scratch, based on **Attention Is All You Need**.

<p align="center">
  <img src="transformer_architecture.jpg" alt="Transformer Architecture" width="350">
</p>

<p align="center">
  <em>Figure. Transformer Architecture</em>
</p>

---

## Overview
- 이 레포지토리는 Vaswani et al.(2017)의 *Attention Is All You Need* 논문을 바탕으로 Transformer의 핵심 구성 요소를 PyTorch로 구현한 코드입니다.

- This repository provides a PyTorch implementation of the core components of the Transformer, based on Attention Is All You Need (Vaswani et al., 2017).
---

## Step 1. DataSet

- 토큰 ID를 고정 차원 벡터(`d_model`)로 변환
- 임베딩 결과에 `sqrt(d_model)` 스케일링 적용

