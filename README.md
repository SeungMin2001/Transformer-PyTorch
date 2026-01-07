# Transformer (PyTorch)
> A PyTorch implementation of the Transformer built from scratch, based on **Attention Is All You Need**.

<br>
<p align="left">
  <img src="assets/transformer_architecture.jpg" alt="Transformer Architecture" width="350">
</p>
<p align="left">
  <em>Figure. Transformer Architecture</em>
</p>


## Paper Reference

This implementation is based on the following paper:

> **Attention Is All You Need**  
> Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,  
> Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
> *Advances in Neural Information Processing Systems (NeurIPS), 2017*

arXiv: https://arxiv.org/abs/1706.03762
<br>

---
## Step 1. Dataset

```python
#data 로드
data=load_dataset("bentrevett/multi30k")

train=data['train']
valid=data['validation']
test=data['test']
train_en=train['en']
train_de=train['de']
```
<br>

---
## Step 2. Tokenizer
I use Hugging Face Transformers – AutoTokenizer

```py
# tokenizer 설정
tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

# 배치학습을 위해 128개씩 배치 세트 들어옴. 그럼 128개에 대한 토큰화 진행 후 반환
def collate_fn(batch):
    texts = [x["en"] for x in batch]
    texts2=[x["de"] for x in batch]

    enc = tokenizer(
        texts,
        padding=True,
        return_tensors="pt",
        return_token_type_ids=False
    )
    dec = tokenizer(
        texts2,
        padding=True,
        return_tensors="pt",
        return_token_type_ids=False
    )
    return enc,dec

# 리스트화
res=[k for k in train]
```
<br>

---
## Step 2. Embedding

<p align="left">
  <img src="assets/embedding.jpg" alt="Transformer Architecture" width="550">
</p>
<br>
embedding + positional encoding
<br>

```py
class TokenEmbedding(nn.Module):
  def __init__(self,vocab_size: int, d_model: int):
    super().__init__()
    self.d_model=d_model
    self.embed=nn.Embedding(vocab_size,d_model)

  def embedding(self,input_ids: torch.Tensor)->torch.Tensor:
    x=self.embed(input_ids) #임베딩
    return x*math.sqrt(self.d_model) #루트 d_model 곱해주기

  def positional_encoding(self,embedding_input_ids:torch.Tensor)->torch.Tensor:
    B,T,d_model=embedding_input_ids.shape

    # 0으로된 T,d_model 2차원 벡터 생성
    pe=torch.zeros(T,d_model,device=embedding_input_ids.device,dtype=embedding_input_ids.dtype)

    pe[:,::2]=torch.sin(T*torch.exp(-torch.div(d_model,512)*math.log(10000)))
    pe[:,1::2]=torch.cos(T*torch.exp(-torch.div(d_model,512)*math.log(10000)))

    # print(embedding_input_ids[0])
    # print(pe[0])

    return embedding_input_ids+pe
```
<br>

---
## Step 3. Attention

<p align="left">
  <img src="assets/attention.png" alt="Transformer Architecture" width="550">
</p>
<br>

attention in encoding, attention in decoding, encoding+decoding attention in decoding <br>
(B,T,d_model) -> (B,T,d_model) same dimention
<br>

```py
class Attention(nn.Module):
  def __init__(self,d_k,d_v,d_model):
    super().__init__()
    self.d_k=d_k
    self.d_v=d_v
    self.d_model=d_model

    self.q_linear_proj=nn.Linear(self.d_model,self.d_model)
    self.k_linear_proj=nn.Linear(self.d_model,self.d_model)
    self.v_linear_proj=nn.Linear(self.d_model,self.d_model)
    self.o_linear_proj=nn.Linear(self.d_model,self.d_model)

    self.dropout=nn.Dropout(p=0.1)
    self.norm=nn.LayerNorm(d_model)

  def forward(self,query,key,value,mask=None): # Q in decoding, K,V in encoding
    Q=self.q_linear_proj(query)
    K=self.k_linear_proj(key)
    V=self.v_linear_proj(value)
    self.B,self.T,self.d_model=query.size() # 차원 다르기 때문에 맞춰줘야함 Q ,(K,V)
    self.B2,self.T2,self.d_model2=value.size()
    #devide d_model
    Q=Q.view(self.B,self.T,8,self.d_k).transpose(1,2)
    K=K.view(self.B2,self.T2,8,self.d_k).transpose(1,2)
    V=V.view(self.B2,self.T2,8,self.d_v).transpose(1,2)

    #scaled dot-product attention
    if mask != None: # mask 존재->masking attention in decorder
      res=res=torch.softmax((Q @ K.transpose(-2,-1)/math.sqrt(self.d_k))+mask, dim=-1) @ V
    else: # None mask-> self-attention in encoder,decoder
      res=torch.softmax(Q @ K.transpose(-2,-1)/math.sqrt(self.d_k), dim=-1) @ V

    # concat
    res=res.transpose(1,2)
    res=res.contiguous().view(self.B,self.T,8*self.d_v)

    # last linear projection
    O=self.o_linear_proj(res)

    #Add & Norm
    return self.norm(query+self.dropout(O))
```
<br>

---

## Step 4. FFW

<p align="left">
  <img src="assets/FFW.png" alt="Transformer Architecture" width="550">
</p><br>

FeedForward 512->2048->512 <br>
I use ReLU of activate function
<br>

```py
class FeedForward(nn.Module):
  def __init__(self,d_model,d_layer):
    super().__init__()
    self.d_model=d_model
    self.d_layer=d_layer

    self.ffn=nn.Sequential(
        nn.Linear(self.d_model,self.d_layer),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(self.d_layer,self.d_model)

    )
    self.dropout=nn.Dropout(p=0.1)
    self.norm=nn.LayerNorm(d_model)

  def forward(self,data):
    x=self.ffn(data)
    x=self.norm(data+self.dropout(x))

    return x
```
<br>

---

## Step 5. modeling
nx=6 in paper <br>
=> encoding * 6, decoding * 6<br>

```py
class transformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.d_model=512
    self.d_layer=2048
    self.out_proj=nn.Linear(self.d_model,tokenizer.vocab_size) # 마지막 선형변환
    self.embedding_data=TokenEmbedding(tokenizer.vocab_size,512) #token
    self.attention=Attention(64,64,512)
    self.ffw=FeedForward(self.d_model,self.d_layer)
    


  def forward(self,input_ids_enc,tgt_in):

    k_encoding=self.embedding_data.embedding(input_ids_enc) #embedding in encoding
    k_decoding=self.embedding_data.embedding(tgt_in) # embedding in decoding
    start_encoding=self.embedding_data.positional_encoding(k_encoding) #token,embedding,positional in encoding
    start_decoding=self.embedding_data.positional_encoding(k_decoding) #token,embedding, positional in decoding

  #------------------ encoder --------------------
    for _ in range(6): #6번 반복 nx=6

      #multi-head self attention in encoding 실행
      x=self.attention.forward(start_encoding,start_encoding,start_encoding)

      start_encoding=self.ffw.forward(x) #encoding end

  #------------------ decoder --------------------
    #print("encoding.size=",start_encoding.size())
    #print("decoding.size=",start_decoding.size())
    for _ in range(6):
      B,T,d_model=start_decoding.size() # decoding T 활용한 masked vector 생성
      mask=torch.ones(T,T,device=input_ids_enc.device)
      mask=torch.triu(mask,diagonal=1)
      mask=mask.masked_fill(mask==1, float('-inf')) # i<j = -inf, i>=j = 0

      # masking attention in decoding
      x=self.attention.forward(start_decoding,start_decoding,start_decoding,mask)

      # encoding+decoding attention
      start_decoding=self.attention.forward(x,start_encoding,start_encoding)

      # FFW
      start_decoding=self.ffw.forward(start_decoding)

    #print(start_decoding[0])

    # Linear in decoding
    last=self.out_proj(start_decoding)
    return last
```
<br>

---

## Step 6. Run

```py
model=transformer()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loader = DataLoader(res, batch_size=128, shuffle=True, collate_fn=collate_fn)
epoch=100
train_loss=[]

optimizer=torch.optim.Adam(
    model.parameters(),
    betas=(0.9,0.98),
    eps=1e-9
    )

vocab_size=tokenizer.vocab_size
criterion=nn.CrossEntropyLoss(ignore_index=0)

for _ in range(epoch):
  model.train()
  total_loss=0
  for enc,dec in loader:


    input_ids_enc = {k: v.to(device) for k, v in enc.items()}
    input_ids_dec = {k: v.to(device) for k, v in dec.items()}
    
    #input_ids_enc = enc["input_ids"] #encoder
    #attention_mask_enc = enc["attention_mask"]

    #input_ids_dec=dec["input_ids"] #decoder
    #attention_mask_dec=dec["attention_mask"]

    tgt_in=input_ids_dec['input_ids'][:,:-1] # eos 제거  학습용 in decoding
    tgt_out=input_ids_dec['input_ids'][:,1:] # bos 제거  loss 전용 in decoding

    logits=model(input_ids_enc['input_ids'],tgt_in)

    loss=criterion(
        logits.reshape(-1,vocab_size),
        tgt_out.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #train_loss.append(loss.item())
    total_loss+=loss.item()

  avg_loss=total_loss/len(loader)
  train_loss.append(avg_loss)

```
<br>

---

## Step 7. result
### traing loss curve is
<p align="center">
  <img src="assets/loss.png" alt="Transformer Architecture">
</p>


