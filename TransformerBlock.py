import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import math
import Constants

class PositionalEncoding(nn.Module):  # 定义位置编码类，继承自nn.Module
    "Implement the PE function."  # 实现位置编码函数

    def __init__(self, d_model, dropout, max_len=800):  # 初始化方法，定义模型维度、dropout概率和最大长度
        super(PositionalEncoding, self).__init__()  # 调用父类的初始化方法
        self.dropout = nn.Dropout(p=dropout)  # 设置dropout层

        # Compute the positional encodings once in log space.  # 计算位置编码
        pe = torch.zeros(max_len, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1).float()  # 生成位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # 计算除数项
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算sin位置编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算cos位置编码
        pe = pe.unsqueeze(0)  # 增加一个维度
        self.register_buffer('pe', pe)  # 注册位置编码为缓冲区

    def forward(self, x):  # 定义前向传播方法
        x = x + self.pe[:, :x.size(1)]  # 添加位置编码到输入
        return self.dropout(x)  # 应用dropout并返回结果

class TransformerBlock(nn.Module):  # 定义Transformer块类，继承自nn.Module

    def __init__(self, input_size, n_heads=2, is_layer_norm=True, attn_dropout=0.1):  # 初始化方法，定义输入大小、头数、是否使用层归一化和注意力dropout概率
        super(TransformerBlock, self).__init__()  # 调用父类的初始化方法
        self.n_heads = n_heads  # 设置头数
        self.d_k = input_size  # 设置键的维度
        self.d_v = input_size  # 设置值的维度

        self.is_layer_norm = is_layer_norm  # 设置是否使用层归一化
        if is_layer_norm:  # 如果使用层归一化
            self.layer_norm = nn.LayerNorm(normalized_shape=input_size)  # 定义层归一化层

        self.pos_encoding = PositionalEncoding(d_model=input_size, dropout=0.5)  # 定义位置编码层

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))  # 定义查询权重参数
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))  # 定义键权重参数
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_v))  # 定义值权重参数

        self.W_o = nn.Parameter(torch.Tensor(self.d_v * n_heads, input_size))  # 定义输出权重参数
        self.linear1 = nn.Linear(input_size, input_size)  # 定义第一个线性层
        self.linear2 = nn.Linear(input_size, input_size)  # 定义第二个线性层

        self.dropout = nn.Dropout(attn_dropout)  # 定义dropout层
        self.__init_weights__()  # 初始化权重参数

    def __init_weights__(self):  # 定义权重初始化方法
        init.xavier_normal_(self.W_q)  # 使用Xavier初始化查询权重
        init.xavier_normal_(self.W_k)  # 使用Xavier初始化键权重
        init.xavier_normal_(self.W_v)  # 使用Xavier初始化值权重
        init.xavier_normal_(self.W_o)  # 使用Xavier初始化输出权重

        init.xavier_normal_(self.linear1.weight)  # 使用Xavier初始化第一个线性层权重
        init.xavier_normal_(self.linear2.weight)  # 使用Xavier初始化第二个线性层权重

    def FFN(self, X):  # 定义前馈神经网络方法
        output = self.linear2(F.relu(self.linear1(X)))  # 应用第一个线性层和ReLU激活函数，然后应用第二个线性层
        output = self.dropout(output)  # 应用dropout
        return output  # 返回输出

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):  # 定义缩放点积注意力方法
        '''
        :param Q: (*, max_q_words, n_heads, input_size)  # 查询张量
        :param K: (*, max_k_words, n_heads, input_size)  # 键张量
        :param V: (*, max_v_words, n_heads, input_size)  # 值张量
        :param mask: (*, max_q_words)  # 掩码张量
        :param episilon:  # 小数值防止除零
        :return:  # 返回注意力输出
        '''
        temperature = self.d_k ** 0.5  # 计算温度

        Q_K = (torch.einsum("bqd,bkd->bqk", Q, K)) / (temperature + episilon)  # 计算查询和键的点积并缩放
        if mask is not None:  # 如果掩码不为空
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))  # 扩展掩码维度
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().to(Constants.device)  # 创建上三角掩码
            mask_ = mask + pad_mask  # 合并掩码
            Q_K = Q_K.masked_fill(mask_, -2**32 + 1)  # 应用掩码

        Q_K_score = F.softmax(Q_K, dim=-1)  # 应用softmax计算注意力分数
        Q_K_score = self.dropout(Q_K_score)  # 应用dropout
        V_att = Q_K_score.bmm(V)  # 计算加权值
        return V_att   # 返回注意力输出

    def multi_head_attention(self, Q, K, V, mask):  # 定义多头注意力方法
        '''
        :param Q:  # 查询张量
        :param K:  # 键张量
        :param V:  # 值张量
        :param mask: (bsz, max_q_words)  # 掩码张量
        :return:  # 返回多头注意力输出
        '''
        bsz, q_len, _ = Q.size()  # 获取查询张量的尺寸
        bsz, k_len, _ = K.size()  # 获取键张量的尺寸
        bsz, v_len, _ = V.size()  # 获取值张量的尺寸

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)  # 计算查询权重并调整维度
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)  # 计算键权重并调整维度
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)  # 计算值权重并调整维度

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)  # 调整查询张量维度
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)  # 调整键张量维度
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)  # 调整值张量维度

        if mask is not None:  # 如果掩码不为空
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)  # 扩展掩码维度
            mask = mask.reshape(-1, mask.size(-1))  # 调整掩码维度

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)  # 计算缩放点积注意力
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)  # 调整注意力输出维度
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)  # 调整注意力输出维度

        output = self.dropout(V_att.matmul(self.W_o))  # 计算输出并应用dropout
        return output   # 返回输出

    def forward(self, Q, K, V, mask=None, pos=True):  # 定义前向传播方法
        '''
        :param Q: (batch_size, max_q_words, input_size)  # 查询张量
        :param K: (batch_size, max_k_words, input_size)  # 键张量
        :param V: (batch_size, max_v_words, input_size)  # 值张量
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q  # 返回输出，尺寸与查询张量相同
        '''
        if pos:  # 如果使用位置编码
            Q = self.pos_encoding(Q)  # 应用位置编码到查询张量
            K = self.pos_encoding(K)  # 应用位置编码到键张量
            V = self.pos_encoding(V)  # 应用位置编码到值张量

        V_att = self.multi_head_attention(Q, K, V, mask)  # 计算多头注意力

        if self.is_layer_norm:  # 如果使用层归一化
            X = self.layer_norm(Q + V_att)  # 应用层归一化到查询张量和注意力输出的和
            output = self.layer_norm(self.FFN(X) + X)  # 应用前馈神经网络和层归一化
        else:  # 如果不使用层归一化
            X = Q + V_att  # 计算查询张量和注意力输出的和
            output = self.FFN(X) + X  # 应用前馈神经网络
        return output  # 返回输出