import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_ce_img import vit_base_patch16_224_ce
from .backbone_txt import build_backbone_roberta

class OSVGCE(nn.Module):
    def __init__(self, args):
        super(OSVGCE, self).__init__()
        print("This is the OSVG model.")
        # img backbone
        self.backbone_img = vit_base_patch16_224_ce(pretrained="pretrained_models/mae_pretrain_vit_base.pth", ce_loc=[3, 6, 9], ce_keep_ratio=[0.7,0.7,0.7])
        self.backbone_img.finetune_vg(img_size=args.imsize, patch_start_index=1)
        # txt backbone
        self.backbone_txt = build_backbone_roberta(args, self.backbone_img.embed_dim)
        hidden_dim = self.backbone_img.embed_dim
        # head
        self.neck = nn.Linear(len(self.backbone_img.return_stage) * hidden_dim, hidden_dim)
        self.head = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, img_data, text_data, ce_keep_rate=None):

        txt_tokens = self.backbone_txt(text_data.tensors.squeeze(), text_data.mask.squeeze())  # [32, 40, 768]
        ml_x, aux_dict = self.backbone_img(x=img_data.tensors, txt_tokens=txt_tokens, ce_keep_rate=ce_keep_rate) # [32, 237, 768]
        x = torch.cat([x[:, 0, :] for x in ml_x ], dim=1)
        x = self.neck(x) # 获取cls_token
        pred_box = self.head(x).sigmoid()
        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


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