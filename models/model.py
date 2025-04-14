import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import GCNLayer, GATLayer, Decoder, Classifier
from models.layers import AttentionLayer, GlobalCode, GlobalVisit, GRULayer, LSTMLayer


class DualMAR(torch.nn.Module):
    def __init__(self, model_config, emb_init, num_classes,
                 use_lab=False, code_fuse='simple', visit_fuse='simple', train_type='direct',
                 lab_weight=None, lab_bias=None, gnn_type='gat', **kwargs):
        super(DualMAR, self).__init__()

        # Graph Learning - GNNs
        input_dim = model_config['init_dim']
        (gnn_hidden, gnn_out), gnn_dropout = model_config['GNN']['dims'], model_config['GNN']['dropout']
        if gnn_type == 'gcn':
            self.gnn_layer = GCNLayer(input_dim, gnn_hidden, gnn_out, emb_init, dropout=gnn_dropout)
        elif gnn_type == 'gat':
            self.gnn_layer = GATLayer(input_dim, gnn_hidden, gnn_out, emb_init, dropout=gnn_dropout)
        else:
            raise ValueError('Invalid GNN type: %s' % gnn_type)

        # Patient Embedding - Attention
        attn_dropout = model_config['Attention']['dropout']
        self.visit_attention = self.retrieve_layer('codes', code_fuse)(gnn_out, dropout=attn_dropout)
        self.patient_attention = self.retrieve_layer('visits', visit_fuse)(gnn_out, dropout=attn_dropout)

        self.use_lab = use_lab
        self.train_type = train_type  # ['direct', 'pretrain', 'finetune']

        # For train_lab
        if self.use_lab:
            self.lab_layer = nn.Linear(290, 256)
            if lab_weight is not None:
                self.lab_layer.weight = nn.Parameter(lab_weight)
                self.lab_layer.weight.requires_grad = False
            if lab_bias is not None:
                self.lab_layer.bias = nn.Parameter(lab_bias)
                self.lab_layer.bias.requires_grad = False

        self.add_dim = 256 if self.use_lab else 0

        decoder_h1, decoder_h2 = model_config['Decoder']['dims']
        decoder_dropout = model_config['Decoder']['dropout']
        classifier_h = model_config['Classifier']['dims'][0]
        classifier_dropout = model_config['Classifier']['dropout']

        # Decoding
        if self.train_type == 'direct':
            self.classifier = Decoder(gnn_out + self.add_dim, decoder_h1, decoder_h2, num_classes[3],
                                      dropout=decoder_dropout)
        else:  # ['pretrain', 'finetune']
            self.decoder1 = Decoder(gnn_out, decoder_h1, decoder_h2, num_classes[0], dropout=decoder_dropout)
            self.decoder2 = Decoder(gnn_out, decoder_h1, decoder_h2, num_classes[1], dropout=decoder_dropout)
            self.decoder3 = Decoder(gnn_out, decoder_h1, decoder_h2, num_classes[2], dropout=decoder_dropout)
            if self.train_type == 'finetune':
                self.classifier = Classifier(gnn_out + decoder_h2 * 3 + self.add_dim, classifier_h, num_classes[3],
                                             dropout=classifier_dropout)

    def forward(self, data, patient_data):
        codes_x = patient_data['codes_x']
        embeddings = self.gnn_layer(data.edge_index, data.edge_weight)

        # For Diagnosis
        patient_embeddings = []
        for patient in range(codes_x.shape[0]):
            visit_embeddings = []
            for visit in range(codes_x.shape[1]):
                visit_codes = codes_x[patient, visit, :]
                visit_codes = visit_codes[visit_codes > 0]
                if len(visit_codes) == 0:
                    continue
                code_embeddings = embeddings[visit_codes]  # code: torch.Size([10, 256])
                visit_embedding = self.visit_attention(code_embeddings)  # visit: torch.Size([256])
                visit_embeddings.append(visit_embedding)

            if len(visit_embeddings) == 0:
                patient_embeddings.append(torch.zeros(self.patient_attention.attn_weights.shape[1]))
            else:
                visit_embeddings = torch.stack(visit_embeddings)
                patient_embedding = self.patient_attention(visit_embeddings)
                patient_embeddings.append(patient_embedding)
        patient_embeddings = torch.stack(patient_embeddings)

        if self.train_type == 'pretrain':
            predictions1 = self.decoder1(patient_embeddings)
            predictions2 = self.decoder2(patient_embeddings)
            predictions3 = self.decoder3(patient_embeddings)
            return predictions1, predictions2, predictions3
        else:  # ['direct', 'finetune']
            if self.train_type == 'finetune':
                lab_result_embeddings1 = self.decoder1.extract_embeddings(patient_embeddings)
                lab_result_embeddings2 = self.decoder2.extract_embeddings(patient_embeddings)
                lab_result_embeddings3 = self.decoder3.extract_embeddings(patient_embeddings)
                patient_embeddings = torch.cat([lab_result_embeddings1, lab_result_embeddings2,
                                                lab_result_embeddings3, patient_embeddings], dim=-1)  #
            if self.use_lab:
                lab_embedding = F.relu(self.lab_layer(patient_data['labs_x']))
                patient_embeddings = torch.cat([patient_embeddings, lab_embedding], dim=-1)

            predictions = self.classifier(patient_embeddings)
            return predictions

    def load_partial_state_dict(self, state_dict):
        model_dict = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'classifier' not in k}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def load_decoder_weights(self, hema_path, chem_path, blood_path, device='cuda:0'):
        self.decoder1.load_state_dict(torch.load(hema_path, map_location=device))
        self.decoder2.load_state_dict(torch.load(chem_path, map_location=device))
        self.decoder3.load_state_dict(torch.load(blood_path, map_location=device))

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
        }
        return layers_dict[module][method]
