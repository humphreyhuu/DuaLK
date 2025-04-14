import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import GCNLayer, Decoder, Classifier
from models.layers import AttentionLayer, GlobalCode, GlobalVisit, GRULayer, LSTMLayer
from models.layers import FusionLayer, ConcatLayer


class LABEHR(torch.nn.Module):
    def __init__(self, gcn_in_channels, gcn_out_channels, emb_init, attn_dim, num_classes, pretrain=False,
                 visit_fuse='simple', code_fuse='simple', direct_train=False, **kwargs):
        super(LABEHR, self).__init__()
        self.gcn_layer = GCNLayer(gcn_in_channels, gcn_out_channels, emb_init)
        self.visit_attention = self.retrieve_layer('codes', code_fuse)(gcn_out_channels, attn_dim)
        self.dense_v2p = nn.Linear(attn_dim, attn_dim)
        self.patient_attention = self.retrieve_layer('visits', visit_fuse)(attn_dim, attn_dim)

        self.direct_train = direct_train
        self.pretrain = pretrain

        if self.direct_train:
            self.classifier = Decoder(attn_dim, num_classes[3])
        else:
            self.decoder1 = Decoder(attn_dim, num_classes[0])
            self.decoder2 = Decoder(attn_dim, num_classes[1])
            self.decoder3 = Decoder(attn_dim, num_classes[2])
            if not self.pretrain:
                self.classifier = Classifier(128 * 3, num_classes[3])

    def forward(self, data, patients_data):
        # patient_data: (batch_size, max_seq_len, max_code_in_a_visit)
        embeddings = self.gcn_layer(data.x, data.edge_index)  # (batch_size, max_seq_len, max_code_in_a_visit, code_dim)
        batch_size, max_seq_len, max_code_in_a_visit = patients_data.shape
        code_embeddings = embeddings[patients_data.view(-1)].view(batch_size, max_seq_len, max_code_in_a_visit, -1)

        mask_code = (patients_data > 0).float()  # (batch_size, max_seq_len, max_code_in_a_visit)
        code_embeddings = code_embeddings * mask_code.unsqueeze(-1)
        visit_embeddings = self.visit_attention(code_embeddings, mask_code)  # (batch_size, max_seq_len, attn_dim)
        visit_embeddings = F.relu(self.dense_v2p(visit_embeddings))

        mask_visit = (patients_data.sum(dim=-1) > 0).float()  # (batch_size, max_seq_len)
        visit_embeddings = visit_embeddings * mask_visit.unsqueeze(-1)
        patient_embeddings = self.patient_attention(visit_embeddings, mask_visit)  # (batch_size, attn_dim)

        if self.direct_train:
            predictions = self.classifier(patient_embeddings)
            return predictions
        else:
            if self.pretrain:
                predictions1 = self.decoder1(patient_embeddings)
                predictions2 = self.decoder2(patient_embeddings)
                predictions3 = self.decoder3(patient_embeddings)
                return predictions1, predictions2, predictions3
            else:
                lab_result_embeddings1 = self.decoder1.extract_embeddings(patient_embeddings)
                lab_result_embeddings2 = self.decoder2.extract_embeddings(patient_embeddings)
                lab_result_embeddings3 = self.decoder3.extract_embeddings(patient_embeddings)
                fused_embedding = torch.cat([lab_result_embeddings1, lab_result_embeddings2,
                                             lab_result_embeddings3], dim=-1)  # 1
                predictions = self.classifier(fused_embedding)
                return predictions

    def load_partial_state_dict(self, state_dict):
        model_dict = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'classifier' not in k}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def load_decoder_weights(self, hema_path, chem_path, blood_path):
        self.decoder1.load_state_dict(torch.load(hema_path))
        self.decoder2.load_state_dict(torch.load(chem_path))
        self.decoder3.load_state_dict(torch.load(blood_path))

    @staticmethod
    def retrieve_layer(module, method):
        layers_dict = {
            'codes': {
                'simple': AttentionLayer,
                'global': GlobalCode,
            },
            'visits': {
                'simple': AttentionLayer,
                'global': GlobalVisit,
                'gru': GRULayer,
                'lstm': LSTMLayer,
            },
            'fusion': {
                'attn': FusionLayer,
                'concat': ConcatLayer,
            }
        }
        return layers_dict[module][method]
