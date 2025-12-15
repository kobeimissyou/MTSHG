import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import Constants
from torch.nn.parameter import Parameter
from TransformerBlock import TransformerBlock

# 多尺度时间特征：引入多尺度时间特征，捕捉不同时间粒度的信息。
class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, hidden_size, num_scales=3, n_heads=8, time_windows=[2, 4, 8]):  # 初始化方法
        super(MultiScaleTemporalAttention, self).__init__()  # 调用父类的初始化方法
        self.time_windows = time_windows  # 设置时间窗口
        self.attentions = nn.ModuleList([  # 定义多头注意力模块列表
            TransformerBlock(hidden_size, n_heads=n_heads) for _ in range(num_scales)  # 每个时间尺度对应一个 TransformerBlock
        ])
        self.num_scales = num_scales  # 设置时间尺度数量
        self.layer_norm = nn.LayerNorm(hidden_size)  # 定义层归一化

    def forward(self, x, mask=None):  # 前向传播方法
        outputs = []  # 存储每个时间尺度的输出
        seq_length = x.size(1)  # 获取输入序列的长度
        for i, window in enumerate(self.time_windows):  # 遍历每个时间窗口
            sliced_x = []  # 存储切片后的序列
            for start in range(0, seq_length, window):  # 按时间窗口切片序列
                end = min(start + window, seq_length)  # 计算切片的结束位置
                sliced_x.append(x[:, start:end, :])  # 添加切片到列表
            sliced_x = torch.cat(sliced_x, dim=1)  # 将切片拼接成一个张量
            att_output = self.attentions[i](sliced_x, sliced_x, sliced_x, mask)  # 输入到对应的 TransformerBlock 中
            padded_output = torch.zeros_like(x)  # 创建与输入相同形状的全零张量
            padded_output[:, :att_output.size(1), :] = att_output  # 将输出填充到原始序列长度
            outputs.append(padded_output)  # 添加填充后的输出到列表

        outputs = torch.stack(outputs)  # 将所有时间尺度的输出堆叠成一个张量
        outputs = torch.mean(outputs, dim=0)  # 对堆叠的张量取平均
        outputs = self.layer_norm(outputs + x)  # 应用层归一化和残差连接
        return outputs  # 返回输出

# 多层次结构特征：引入多层次结构特征，捕捉不同层次的结构信息。
class MultiLevelStructuralAttention(nn.Module):
    def __init__(self, hidden_size, num_levels=3, n_heads=8, structure_windows=[2, 4, 8]):  # 初始化方法
        super(MultiLevelStructuralAttention, self).__init__()  # 调用父类的初始化方法
        self.structure_windows = structure_windows  # 各层次切片窗口大小
        self.attentions = nn.ModuleList([  # 定义多层次的 TransformerBlock 模块
            TransformerBlock(hidden_size, n_heads=n_heads) for _ in range(num_levels) # 每个层次对应一个 TransformerBlock
        ])
        self.num_levels = num_levels  # 设置层次数量
        self.layer_norm = nn.LayerNorm(hidden_size)  # 定义层归一化

    def forward(self, x, mask=None):  # 前向传播方法
        batch_size, seq_length, _ = x.size()  # 获取输入张量的批量大小、序列长度和特征维度
        outputs = []  # 存储每个层次的输出
        for i, window in enumerate(self.structure_windows):  # 遍历每个切片窗口
            sliced_segments = []  # 存储切片后的序列
            for start in range(0, seq_length, window):  # 按窗口大小切片序列
                end = min(start + window, seq_length)  # 防止越界
                segment = x[:, start:end, :]  # 切片
                sliced_segments.append(segment)  # 添加切片到列表
            sliced_x = torch.cat(sliced_segments, dim=1)  # 拼接切片后的序列
            att_mask = mask[:, :sliced_x.size(1)] if mask is not None else None  # 生成对应切片的掩码
            level_output = self.attentions[i](sliced_x, sliced_x, sliced_x, att_mask)  # 输入到对应层次的 TransformerBlock 处理
            padded_output = torch.zeros_like(x)  # 初始化全零张量
            padded_output[:, :level_output.size(1), :] = level_output  # 填充有效部分到原始序列长度
            outputs.append(padded_output)  # 添加填充后的输出到列表

        outputs = torch.stack(outputs, dim=0)  # 堆叠所有层次的输出
        outputs = torch.mean(outputs, dim=0)  # 对堆叠的张量取平均
        outputs = self.layer_norm(outputs + x)  # 残差连接 + 层归一化
        return outputs  # 返回最终输出

# 门控融合：引入多头注意力机制、归一化技术、残差连接来处理输入的两个特征向量，然后进行线性变换和融合。
class Gated_fusion(nn.Module):
    def __init__(self, input_size, out_size=1, dropout=0.3, num_heads=6):
        super(Gated_fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)  # 线性层，用于变换输入特征
        self.linear2 = nn.Linear(input_size, out_size)  # 线性层，用于变换特征到输出大小
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)  # 多头注意力层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.layer_norm1 = nn.LayerNorm(input_size)  # 层归一化
        self.layer_norm2 = nn.LayerNorm(input_size)  # 层归一化
        self.gelu = nn.GELU()  # GELU激活函数
        self.init_weights()  # 初始化权重

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)  # 初始化linear1的权重
        init.xavier_normal_(self.linear2.weight)  # 初始化linear2的权重

    def forward(self, X1, X2):
        concatenated_features = torch.cat([X1.unsqueeze(dim=0), X2.unsqueeze(dim=0)], dim=0)  # 拼接输入特征
        attn_output, _ = self.attention(concatenated_features, concatenated_features, concatenated_features)  # 应用多头注意力
        attn_output = self.dropout(attn_output)  # 应用Dropout
        attn_output = self.layer_norm1(attn_output + concatenated_features)  # 应用层归一化和残差连接

        transformed_output = self.linear1(attn_output)  # 使用linear1变换特征
        transformed_output = self.gelu(transformed_output)  # 应用GELU激活
        transformed_output = self.dropout(transformed_output)  # 应用Dropout
        transformed_output = self.layer_norm2(transformed_output + attn_output)  # 应用层归一化和残差连接

        emb_score = F.softmax(self.linear2(transformed_output), dim=0)  # 对变换后的特征应用softmax
        emb_score = self.dropout(emb_score)  # 应用Dropout

        output = torch.sum(emb_score * concatenated_features, dim=0)  # 计算拼接特征的加权和
        return output  # 返回输出

# MTSHG模型：结合多尺度时间特征、多层次结构特征的模型。
class MTSHG(nn.Module):  # 定义MTSHG类，继承自nn.Module
    def __init__(self, opt):  # 初始化方法，定义模型参数
        super(MTSHG, self).__init__()  # 调用父类的初始化方法

        self.hidden_size = opt.d_model  # 设置隐藏层大小
        self.n_node = opt.user_size  # 设置节点数量
        self.dropout = nn.Dropout(opt.dropout)  # 定义dropout层
        self.initial_feature = opt.initialFeatureSize  # 设置初始特征大小
        self.hgnn = HGNN(self.initial_feature, self.hidden_size, dropout=opt.dropout)  # 定义HGNN模型

        self.user_embedding = nn.Embedding(self.n_node, self.initial_feature)  # 定义用户嵌入层
        self.stru_attention = MultiLevelStructuralAttention(self.hidden_size)  # 定义多层次结构注意力机制
        self.temp_attention = MultiScaleTemporalAttention(self.hidden_size)  # 定义多尺度时间注意力机制

        self.global_cen_embedding = nn.Embedding(600, self.hidden_size)  # 定义全局中心嵌入层
        self.local_time_embedding = nn.Embedding(5000, self.hidden_size)  # 定义本地时间嵌入层
        self.cas_pos_embedding = nn.Embedding(50, self.hidden_size)  # 定义级联位置嵌入层
        self.local_inf_embedding = nn.Embedding(200, self.hidden_size)  # 定义本地影响嵌入层

        self.weight = Parameter(torch.Tensor(self.hidden_size + 2, self.hidden_size + 2))  # 定义权重参数
        self.weight2 = Parameter(torch.Tensor(self.hidden_size + 2, self.hidden_size + 2))  # 定义第二个权重参数
        self.fus = Gated_fusion(self.hidden_size + 2)  # 定义门控融合层
        self.linear = nn.Linear((self.hidden_size + 2), 2)  # 定义线性层
        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):  # 定义初始化参数方法
        stdv = 1.0 / math.sqrt(self.hidden_size)  # 计算标准差
        for weight in self.parameters():  # 遍历所有参数
            weight.data.uniform_(-stdv, stdv)  # 初始化参数

    def forward(self, data_idx, hypergraph_list):  # 定义前向传播方法
        news_centered_graph, user_centered_graph, spread_status = (item for item in hypergraph_list)  # 解包超图列表
        seq, timestamps, user_level = (item for item in news_centered_graph)  # 解包新闻中心图
        useq, user_inf, user_cen = (item for item in user_centered_graph)  # 解包用户中心图

        hidden = self.dropout(self.user_embedding.weight)  # 获取用户嵌入并应用dropout
        user_cen = self.global_cen_embedding(user_cen)  # 获取全局中心嵌入
        tweet_hidden = hidden + user_cen  # 计算推文隐藏层
        user_hgnn_out = self.hgnn(tweet_hidden, seq, useq)  # 计算HGNN输出

        zero_vec1 = -9e15 * torch.ones_like(seq[data_idx])  # 创建全零向量
        one_vec = torch.ones_like(seq[data_idx], dtype=torch.float)  # 创建全一向量
        nor_input = torch.where(seq[data_idx] > 0, one_vec, zero_vec1)  # 计算归一化输入
        nor_input = F.softmax(nor_input, 1)  # 应用softmax
        att_mask = (seq[data_idx] == Constants.PAD)  # 计算注意力掩码
        adj_with_fea = F.embedding(seq[data_idx], user_hgnn_out)  # 计算嵌入特征

        global_time = self.local_time_embedding(timestamps[data_idx])  # 获取全局时间嵌入
        att_hidden = adj_with_fea + global_time  # 计算注意力隐藏层

        att_out = self.temp_attention(att_hidden, mask=att_mask)  # 计算时间注意力输出
        news_out = torch.einsum("abc,ab->ac", (att_out, nor_input))  # 计算新闻输出

        news_out = torch.cat([news_out, spread_status[data_idx][:, 2:] / 3600 / 24], dim=-1)  # 拼接新闻输出
        news_out = news_out.matmul(self.weight)  # 计算新闻输出

        local_inf = self.local_inf_embedding(user_inf[data_idx])  # 获取本地影响嵌入
        cas_pos = self.cas_pos_embedding(user_level[data_idx])  # 获取级联位置嵌入
        att_hidden_str = adj_with_fea + local_inf + cas_pos  # 计算结构注意力隐藏层

        att_out_str = self.stru_attention(att_hidden_str, mask=att_mask)  # 计算结构注意力输出
        news_out_str = torch.einsum("abc,ab->ac", (att_out_str, nor_input))  # 计算结构新闻输出

        news_out_str = torch.cat([news_out_str, spread_status[data_idx][:, :2]], dim=-1)  # 拼接结构新闻输出
        news_out_str = news_out_str.matmul(self.weight2)  # 计算结构新闻输出

        news_out = self.fus(news_out, news_out_str)  # 计算门控融合输出
        output = self.linear(news_out)  # 计算线性输出
        output = F.log_softmax(output, dim=1)  # 应用log_softmax

        return output  # 返回输出

# HGNN层：定义HGNN的单层结构。
class HGNN_layer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.3):
        super(HGNN_layer, self).__init__()
        self.dropout = dropout  # 设置dropout概率
        self.in_features = input_size  # 输入特征的大小
        self.out_features = output_size  # 输出特征的大小
        self.weight1 = Parameter(torch.Tensor(self.in_features, self.out_features))  # 定义第一个权重参数
        self.weight2 = Parameter(torch.Tensor(self.out_features, self.out_features))  # 定义第二个权重参数
        self.att_linear = nn.Linear(self.out_features, 1)
        self.reset_parameters()  # 初始化权重参数

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)  # 计算标准差
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)  # 初始化参数
        self.weight1.data.uniform_(-stdv, stdv)  # 初始化第一个权重参数
        self.weight2.data.uniform_(-stdv, stdv)  # 初始化第二个权重参数

    def forward(self, x, seq, useq):
        x = x.matmul(self.weight1)  # 计算节点特征
        adj_with_fea = F.embedding(seq, x)  # 计算嵌入特征

        att_scores = self.att_linear(adj_with_fea).squeeze(-1)  # 计算注意力分数
        zero_vec1 = -9e15 * torch.ones_like(seq)  # 创建全零向量
        one_vec = torch.ones_like(seq, dtype=torch.float)  # 创建全一向量
        att_mask = torch.where(seq > 0, one_vec, zero_vec1)  # 计算注意力掩码
        att_scores = att_scores + att_mask  # 将掩码添加到注意力分数中
        att_weights = F.softmax(att_scores, dim=1).unsqueeze(-1)  # 计算注意力权重
        edge = torch.sum(att_weights * adj_with_fea, dim=1)  # 计算边特征

        zero_vec1 = -9e15 * torch.ones_like(seq)  # 创建全零向量
        one_vec = torch.ones_like(seq, dtype=torch.float)  # 创建全一向量
        nor_input = torch.where(seq > 0, one_vec, zero_vec1)  # 计算归一化输入
        nor_input = F.softmax(nor_input, 1)  # 应用softmax

        edge = torch.einsum("abc,ab->ac", (adj_with_fea, nor_input))  # 计算边特征
        edge = F.dropout(edge, self.dropout, training=self.training)  # 应用dropout
        edge = F.relu(edge, inplace=False)  # 应用ReLU激活函数
        e1 = edge.matmul(self.weight2)  # 计算边特征
        edge_adj_with_fea = F.embedding(useq, e1)  # 计算嵌入特征

        zero_vec1 = -9e15 * torch.ones_like(useq)  # 创建全零向量
        one_vec = torch.ones_like(useq, dtype=torch.float)  # 创建全一向量
        u_nor_input = torch.where(useq > 0, one_vec, zero_vec1)  # 计算归一化输入
        u_nor_input = F.softmax(u_nor_input, 1)  # 应用softmax
        node = torch.einsum("abc,ab->ac", (edge_adj_with_fea, u_nor_input))  # 计算节点特征

        node = F.dropout(node, self.dropout, training=self.training)  # 应用dropout
        return node  # 返回节点特征

# HGNN模型：定义HGNN的整体结构
class HGNN(nn.Module):
    def __init__(self, input_size, output_size, scales=8, dropout=0.3):
        super(HGNN, self).__init__()
        self.scales = scales
        self.dropout = dropout
        # 计算均匀划分比例
        self.ratios = [1.0 / scales] * scales
        # 初始化各尺度的HGNN层
        self.gnn_layers = nn.ModuleList([
            HGNN_layer(input_size, output_size, dropout=dropout)
            for _ in range(scales)
        ])

    def forward(self, x, seq, useq):
        outputs = []
        seq_length = seq.size(1)
        useq_length = useq.size(1)
        # 序列起始位置
        seq_start, useq_start = 0, 0

        for scale in range(self.scales):
            # 计算当前尺度的序列长度（均匀划分）
            seq_end = int(seq_start + seq_length * self.ratios[scale])
            useq_end = int(useq_start + useq_length * self.ratios[scale])
            # 确保最后一个片段包含剩余的所有元素
            if scale == self.scales - 1:
                seq_end = seq_length
                useq_end = useq_length
            # 提取子序列
            sub_seq = seq[:, seq_start:seq_end]
            sub_useq = useq[:, useq_start:useq_end]
            # 应用当前尺度的HGNN层
            output = self.gnn_layers[scale](x, sub_seq, sub_useq)
            outputs.append(output)
            # 更新起始位置
            seq_start, useq_start = seq_end, useq_end

        # 融合多尺度输出
        node = torch.mean(torch.stack(outputs), dim=0)
        return node