import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * rms)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
        self.do = nn.Dropout(dropout)
        self.norm = RMSNorm(hidden_size)

    def forward(self, q, k, v, mask=None, return_attention=False):
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32))

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.do(attention_weights)

        attention_output = torch.matmul(attention_weights, V)
        output = self.fc(attention_output)
        output = self.norm(output + q)

        if return_attention:
            return output, attention_weights
        else:
            return output

class SentenceLevelGlobalAttention(nn.Module):
    def __init__(self, text_dim=768, dropout=0.5):
        super(SentenceLevelGlobalAttention, self).__init__()
        # 第一个多头注意力层
        self.attention1 = nn.MultiheadAttention(embed_dim=text_dim, num_heads=8, dropout=dropout)
        self.norm1 = RMSNorm(text_dim)

        # 引入可学习的权重向量
        self.weights = nn.Parameter(torch.ones(text_dim))

    def forward(self, text_embedding):
        # 对于xlnet来说，此时text_embedding 的形状是 (seq_len, batch_size, text_dim)

        # 计算加权平均，得到查询向量
        weighted_text_embedding = text_embedding * self.weights.unsqueeze(0).unsqueeze(1)
        query = torch.sum(weighted_text_embedding, dim=0, keepdim=True) / torch.sum(self.weights)

        # 第一个多头注意力层的计算
        global_context1, _ = self.attention1(query, text_embedding, text_embedding)
        global_context1 = self.norm1(global_context1 + text_embedding)  # (1, batch_size, text_dim)

        return global_context1


class GCAA(nn.Module):
    def __init__(self, text_dim=768, dropout=0.5, hidden_size_gru=256):
        super(GCAA, self).__init__()

        # # 使用GRU代替原始的Embedding层进行特征提取
        self.visual_gru = nn.GRU(input_size=47, hidden_size=hidden_size_gru, batch_first=True, bidirectional=True)
        self.acoustic_gru = nn.GRU(input_size=74, hidden_size=hidden_size_gru, batch_first=True, bidirectional=True)

        # GRU输出维度调整以匹配Transformer的输入维度
        self.visual_linear = nn.Linear(hidden_size_gru * 2, text_dim)
        self.acoustic_linear = nn.Linear(hidden_size_gru * 2, text_dim)

        self.cross_modal_attention = CrossAttention(text_dim, dropout=dropout)

        self.cat_connect = nn.Linear(2 * text_dim, text_dim)
        self.norm = RMSNorm(text_dim)

        # 新增基于句子级别的全局上下文建模模块
        self.global_context_module = SentenceLevelGlobalAttention(text_dim, dropout)

    def forward(self, text_embedding, visual_input, acoustic_input):
        # 获取句子级别的全局上下文表示
        global_context = self.global_context_module(text_embedding)

        # 将全局上下文表示广播到每个token的位置上，并与文本嵌入相加
        global_context_expanded = global_context.expand(text_embedding.size(0), -1, -1)
        # global_context_expanded = global_context.unsqueeze(1).expand(text_embedding.size(0), -1, -1)

        text_embedding_with_global_context = text_embedding + global_context_expanded

        # GRU处理视觉和听觉输入
        visual_output, _ = self.visual_gru(visual_input)
        acoustic_output, _ = self.acoustic_gru(acoustic_input)

        # 将GRU输出转换为与文本相同的维度
        visual_encoded = self.visual_linear(visual_output)
        acoustic_encoded = self.acoustic_linear(acoustic_output)

        # 跨模态交互，使用增强后的文本嵌入
        cross_text_visual = self.cross_modal_attention(
            q=text_embedding_with_global_context,
            k=visual_encoded,
            v=visual_encoded
        )

        cross_text_acoustic = self.cross_modal_attention(
            q=text_embedding_with_global_context,
            k=acoustic_encoded,
            v=acoustic_encoded
        )

        # 初始化组合表示
        combined = [cross_text_visual, cross_text_acoustic]

        # 合并所有表示
        combined_tensor = torch.cat(combined, dim=-1)

        # 线性变换以适应最终输出维度
        shift = self.cat_connect(combined_tensor)

        # normalization
        output = self.norm(shift + text_embedding)  # multimodal

        return output, text_embedding_with_global_context, acoustic_encoded, visual_encoded


