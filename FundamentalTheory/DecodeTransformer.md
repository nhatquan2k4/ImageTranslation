# Decoder Transformer

Tài liệu này trình bày chi tiết về Decoder trong kiến trúc Transformer, bao gồm:
- Kiến trúc lớp (Masked Self-Attention, Cross-Attention, FFN)
- Masking (causal/look-ahead, padding, kết hợp mask)
- Mục tiêu huấn luyện (teacher forcing, shift-right, label smoothing)
- Suy luận (greedy, beam search, sampling), KV cache
- Positional encodings (Absolute/Relative/RoPE/ALiBi)


---

<img src="img/decoder.jfif" alt="decoder picture">

---

## 1) Tổng quan Decoder Transformer

- Mục đích: sinh ra mã chuỗi đích Y theo kiểu tự hồi quy (autoregressive) đồng thời liên tục kết hợp ngữ cảnh từ thông tin mã hóa của encoder (các token đã sinh).
- Ứng dụng:
  - Dịch máy, tóm tắt, sinh văn bản, hội thoại, sinh mã, các tác vụ điều kiện theo đầu vào (với encoder) hoặc không (decoder-only).

---

## 2) Kiến trúc một Decoder Layer 
<img src="img/decoder1.jfif">

- Mỗi decoder block có cấu trúc như sau:

<img src="img/decoder2.jfif">

Gọi 𝑋 là đầu vào của khối Decoder (chuỗi embedding từ bước trước hoặc từ embedding ban đầu của từ dịch).

Gọi 𝐻 là output từ Encoder, nó cung cấp ngữ cảnh của câu nguồn.
### 1. Self-Attention (Masked Multi-Head Attention)
$$
\tilde{X} = \text{LayerNorm}(X)
$$
$$
X' = X + \text{Attention}(\tilde{X}, \tilde{X}, \tilde{X})
$$

---

### 2. Cross-Attention (Encoder–Decoder Attention)
$$
\tilde{X}' = \text{LayerNorm}(X')
$$
$$
X'' = X' + \text{Attention}(\tilde{X}', H, H)
$$

Trong đó:  
- \(H\) là **encoder output** (ngữ cảnh từ encoder).  
- \(X'\) là đầu ra sau self-attention.  

---

### 3. Feed Forward Network (FFN)
$$
\tilde{X}'' = \text{LayerNorm}(X'')
$$
$$
X''' = X'' + \text{FFN}(\tilde{X}'')
$$

---

### 4. Output
$$
Y = \text{LayerNorm}(X''')
$$

---

### Công thức gọn cho toàn bộ Decoder block
$$
Y = \text{LayerNorm}\Big( X'' + \text{FFN}(\text{LayerNorm}(X'' )) \Big)
$$

Với:
$$
X'' = X' + \text{Attention}(\text{LayerNorm}(X'), H, H)
$$
$$
X' = X + \text{Attention}(\text{LayerNorm}(X), \text{LayerNorm}(X), \text{LayerNorm}(X))
$$

---

## 3) Self-Attention và Causal Mask

<img src="img/self_attention.jfif">

### Scaled Dot-Product Attention cho Self-Attention

1. Tính Q, K, V:
$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

2. Tính điểm tương đồng (scores):
$$
\text{Scores} = \frac{QK^T}{\sqrt{d_k}}, \quad \text{shape: } (B, h, T, T)
$$

3. Áp dụng mask:
- **Causal mask** $(M_{\text{causal}}$) (tam giác dưới):  
$$
M[i,j] = -\infty \quad \text{nếu } j > i
$$

- **Target padding mask** $(M_{\text{pad}}$):  
Che các vị trí bị padding (thường che cột \(j\) tương ứng token pad).

- Kết hợp:
$$
\text{Scores} = \text{Scores} + M_{\text{causal}} + M_{\text{pad}}
$$

4. Tính Attention:
$$
A = \text{softmax}(\text{Scores})
$$

5. Đầu ra:
$$
\text{Out} = A V
$$

---

**Trực giác**: Ở thời điểm \(t\), một token chỉ được "nhìn" các token ở vị trí $(\leq t$).


---

## 4) Cross-Attention (Encoder-Decoder Attention)

<img src="img/cross_attention.jfif">

### Scaled Dot-Product Attention cho Cross-Attention (Encoder–Decoder Attention)

1. Tính Q, K, V:
$$
Q = X_{\text{dec}} W_Q, \quad K = H_{\text{enc}} W_K, \quad V = H_{\text{enc}} W_V
$$

2. Tính điểm tương đồng (scores):
$$
\text{Scores} = \frac{QK^T}{\sqrt{d_k}}, \quad \text{shape: } (B, h, T_{\text{tgt}}, S_{\text{src}})
$$

- $(B$): batch size  
- $(h$): số head  
- $(T_{\text{tgt}}$): độ dài chuỗi đích (decoder input)  
- $(S_{\text{src}}$): độ dài chuỗi nguồn (encoder input)  

3. Áp dụng **encoder padding mask**:  
Che các cột tương ứng với token bị padding trong câu nguồn.

4. Tính Attention:
$$
A = \text{softmax}(\text{Scores})
$$

5. Đầu ra:
$$
\text{Out} = A V
$$

---

**Ý nghĩa**: Decoder truy xuất thông tin ngữ cảnh từ câu nguồn đã được Encoder mã hóa.

---

## 5) Mục tiêu huấn luyện (Teacher Forcing)

- Dịch trái (shift-right): thêm token $<bos>$ ở đầu chuỗi đích, dịch toàn bộ 1 bước để dự đoán token kế tiếp.
- Loss: cross-entropy giữa logits ở thời t và token thật ở t.
- Mask loss cho các vị trí $<pad>$.
- Label smoothing (ví dụ ε=0.1) giúp tổng quát tốt hơn (nhất là seq2seq).
- Thường “tied embeddings”: dùng chung ma trận embedding và projection tới vocab (giảm tham số, cải thiện chất lượng nhẹ).

---
