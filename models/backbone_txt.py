import torch
from torch import nn
from transformers import RobertaModel,BertModel
from transformers.models.bert.modeling_bert import ACT2FN

class FeatureResizer(nn.Module):
    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class BertAdapter(nn.Module):
    def __init__(self, hidden_size=768, adapter_size=64, adapter_initializer_range=1e-2, adapter_act='relu'):
        super(BertAdapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        nn.init.normal_(self.down_project.weight, std=adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)

        if isinstance(adapter_act, str):
            self.activation = ACT2FN[adapter_act]
        else:
            self.activation = adapter_act

        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)
        nn.init.normal_(self.up_project.weight, std=adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states: torch.Tensor):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return self.norm(hidden_states + up_projected)


class BackboneTxt(nn.Module):
    def __init__(self, text_encoder, hidden_dim,freeze_text_encoder=True):
        super(BackboneTxt, self).__init__()

        self.text_encoder = text_encoder
        # self.adapters=nn.Sequential(*[BertAdapter(hidden_size=hidden_dim) for i in range(12)])
        self.txt_proj = FeatureResizer(
            input_feat_size=self.text_encoder.config.hidden_size,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )
        # if freeze_text_encoder:
        #     for p in self.text_encoder.parameters():
        #         p.requires_grad_(False)

    def forward(self, text_ids, text_masks):
        text_embeds = self.text_encoder.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_encoder.get_extended_attention_mask(text_masks, input_shape, device)

        output=[]
        # extract text feature
        for layer in self.text_encoder.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.txt_proj(text_embeds)
        # extract text feature adapter
        # for adapter in self.adapters:
        #     text_embeds = adapter(text_embeds)
        #     output.append(text_embeds)
        return text_embeds


def build_backbone_roberta(cfg, hidden_dim):
    text_encoder=RobertaModel.from_pretrained("pretrained_models/models-FacebookAI-roberta-base")
    model = BackboneTxt(text_encoder,hidden_dim)
    return model

def build_backbone_bert(cfg, hidden_dim):
    text_encoder=BertModel.from_pretrained("pretrained_models/models-google-bert-bert-base-uncased")
    model = BackboneTxt(text_encoder,hidden_dim)
    return model