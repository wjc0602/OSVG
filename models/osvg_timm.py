import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone_img_sl import vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224
from .backbone_txt import build_backbone_bert, build_backbone_roberta

class OSVG(nn.Module):
    def __init__(self, args):
        super(OSVG, self).__init__()
        print("This is the OSVG model.")
        
        # img backbone
        if (args.img_model == "vitb"):
            print("init ViT-B")
            self.backbone_img = vit_base_patch16_224(pretrained="pretrained_models/mae_pretrain_vit_base.pth")
        elif (args.img_model == "vitl"):
            print("init ViT-L")
            self.backbone_img = vit_large_patch16_224(pretrained="pretrained_models/mae_pretrain_vit_large.pth")
        elif (args.img_model == "vith"):
            print("init ViT-H")
            self.backbone_img = vit_huge_patch14_224(pretrained="pretrained_models/mae_pretrain_vit_large.pth")
        self.backbone_img.finetune_vg(img_size=args.imsize, patch_start_index=1)

        # txt backbone
        if (args.txt_model == "bert"):
            print("init ViT-B")
            self.backbone_txt = build_backbone_bert(args, self.backbone_img.embed_dim)
        elif (args.txt_model == "roberta"):
            print("init ViT-L")
            self.backbone_txt = build_backbone_roberta(args, self.backbone_img.embed_dim)
        hidden_dim = self.backbone_img.embed_dim
        # head
        self.head = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, img_data, text_data):

        txt_tokens = self.backbone_txt(text_data.tensors.squeeze(), text_data.mask.squeeze())  # [32, 40, 768]
        x, aux_dict = self.backbone_img(x=img_data.tensors, txt_tokens=txt_tokens) # [32, 237, 768]
        pred_box = self.head(x[:, 0, :]).sigmoid()
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