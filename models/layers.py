import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 emb_init, dropout=0.2, trainable_emb=True):  # 0.2
        super(GCNLayer, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.node_features = nn.Parameter(emb_init) if trainable_emb else emb_init

    def forward(self, edge_index, edge_weight=None):
        x = self.node_features
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class GATLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 emb_init, dropout=0., trainable_emb=True):  # 0.2
        super(GATLayer, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.node_features = nn.Parameter(emb_init) if trainable_emb else emb_init

    def forward(self, edge_index, edge_weight=None):
        x = self.node_features
        edge_attr = edge_weight.view(-1, 1) if edge_weight is not None else None
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x


class AttentionLayer(torch.nn.Module):
    def __init__(self, attn_dim, dropout=0.2):
        super(AttentionLayer, self).__init__()
        self.attn_weights = torch.nn.Parameter(torch.randn(attn_dim, attn_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention=False):
        scores = torch.matmul(x, self.attn_weights)  # x: (10, 256)
        attn_weights = F.softmax(scores, dim=-2)  # 0
        attn_weights = self.dropout(attn_weights)
        context_vector = torch.sum(attn_weights * x, dim=-2)  # 0
        if return_attention:
            return context_vector, attn_weights
        return context_vector


def masked_softmax(vec, mask):
    vec = vec - vec.max(dim=-1, keepdim=True)[0]
    exp = torch.exp(vec) * mask
    sum_exp = exp.sum(dim=-1, keepdim=True)
    return exp / (sum_exp + 1e-10)


class GlobalCode(nn.Module):
    def __init__(self, attn_dim):
        super(GlobalCode, self).__init__()
        self.attention_size = attn_dim
        self.w_omega = nn.Parameter(torch.randn(attn_dim, attn_dim))
        self.b_omega = nn.Parameter(torch.randn(attn_dim))
        self.u_omega = nn.Parameter(torch.randn(attn_dim))

    def forward(self, x, mask=None):
        v = torch.tanh(torch.matmul(x, self.w_omega) + self.b_omega)  # (batch_size, seq_len, attention_size)
        vu = torch.matmul(v, self.u_omega)  # (batch_size, seq_len)
        if mask is not None:
            vu = vu * mask
            alphas = masked_softmax(vu, mask)
        else:
            alphas = F.softmax(vu, dim=-1)  # (batch_size, seq_len)
        output = torch.sum(x * alphas.unsqueeze(-1), dim=-2)  # (batch_size, input_dim)
        return output  # , alphas


class GlobalVisit(GlobalCode):
    def __init__(self, attn_dim):  # output_dim
        super(GlobalVisit, self).__init__(attn_dim)
        self.u_omega_o = nn.Parameter(torch.randn(attn_dim, attn_dim))  # output_dim

    def forward(self, x, mask=None):
        t = F.normalize(torch.matmul(x, self.w_omega) + self.b_omega, p=2, dim=-1)
        v = torch.tanh(t)  # (batch_size, seq_len, attention_size)
        vu = torch.matmul(v, self.u_omega)  # (batch_size, seq_len)
        vu_o = torch.matmul(v, self.u_omega_o)  # (batch_size, seq_len, output_dim)
        if mask is not None:
            vu = vu * mask
            mask_o = mask.unsqueeze(-1)
            vu_o = vu_o * mask_o
            alphas = masked_softmax(vu, mask)
            betas = masked_softmax(vu_o, mask_o)
        else:
            alphas = F.softmax(vu, dim=-1)  # (batch_size, seq_len)
            betas = F.softmax(vu_o, dim=-2)  # (batch_size, seq_len, output_dim)
        w = alphas.unsqueeze(-1) * betas
        output = torch.sum(x * w, dim=-2)  # (batch_size, input_dim)
        return output  # , alphas, betas


class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x, mask=None):
        output, _ = self.gru(x)
        return output[:, -1, :]


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x, mask=None):
        output, (h_n, c_n) = self.lstm(x)
        return output[:, -1, :]


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim, dropout=0.2):  # 0.4, 0.1
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

    def extract_embeddings(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.relu(self.fc2(x))


class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
