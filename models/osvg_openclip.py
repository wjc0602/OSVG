import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from utils.misc import NestedTensor

def resize_pos_embed(clip_visual, patch_size, imsize_new):

    # for patch embedding
    cls_pos_embed = clip_visual.positional_embedding[:1, :]
    patch_pos_embed_old = clip_visual.positional_embedding[1:, :]
    patch_pos_embed_old = patch_pos_embed_old.transpose(0, 1)
    E, Q = patch_pos_embed_old.shape
    P_H, P_W = clip_visual.input_resolution // patch_size, clip_visual.input_resolution // patch_size
    patch_pos_embed_old = patch_pos_embed_old.view(E, P_H, P_W).unsqueeze(0)

     # for search region
    H, W = imsize_new, imsize_new
    new_P_H, new_P_W = H // patch_size, W // patch_size
    patch_pos_embed_new = nn.functional.interpolate(patch_pos_embed_old, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
    patch_pos_embed_new = patch_pos_embed_new.flatten(2).transpose(1, 2).squeeze(0)
    patch_pos_embed_new = torch.cat([cls_pos_embed, patch_pos_embed_new], dim=0)
    patch_pos_embed_new = nn.Parameter(patch_pos_embed_new)

        # for cls token
    clip_visual.positional_embedding = patch_pos_embed_new

    return clip_visual

class MultiLevel_Transformer(nn.Module):
    def __init__(self, clip_vit, extract_layer):
        super().__init__()
        heads = clip_vit.width // 64
        self.width = clip_vit.width
        self.layers = clip_vit.layers
        self.resblocks = clip_vit.resblocks
        self.extract_layer = extract_layer

    def forward(self, x: torch.Tensor):
        ml_feature = []
        for i in range(max(self.extract_layer)+1):
            x = self.resblocks[i](x)
            if i in self.extract_layer:
                ml_feature.append(x)
        return ml_feature

class MultiLevel_ImageEncoder_modified(nn.Module):
    def __init__(self, clip_visu_model, extract_layer):
        super().__init__()
        self.input_resolution = clip_visu_model.image_size
        self.output_dim = clip_visu_model.output_dim
        self.conv1 = clip_visu_model.conv1
        self.class_embedding = clip_visu_model.class_embedding
        self.positional_embedding = clip_visu_model.positional_embedding
        self.ln_pre = clip_visu_model.ln_pre
        self.transformer = MultiLevel_Transformer(clip_visu_model.transformer, extract_layer)
        self.ln_post = clip_visu_model.ln_post
        self.proj = clip_visu_model.proj
        self.positional_embedding.requires_grad_(True)

    def forward(self, x, text_tensors):
        txt_len = text_tensors.shape[1]
        x = self.conv1(x) # [32, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [32, 768, 196]
        x = x.permute(0, 2, 1) # [32, 196, 768]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # [32, 197, 768]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x) # LN
        x = torch.cat([x, text_tensors], dim=1)
        x = x.permute(1, 0, 2)  # NLD -> LND, B L H -> L B H [197, 32, 768]
        ml_x = self.transformer(x)
        ml_x = [x[:-txt_len, :, :] for x in ml_x] # split txt features
        x = torch.cat(ml_x, dim=2) # [197, 32, 768*4]
        x = x.permute(1, 0, 2)  # LND -> NLD, L B H -> B 4*L H [32, 197, 9072]
        return x

class TextEncoder_modified(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.positional_embedding.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, x):
        x = self.token_embedding(x).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # TODO: ModifiedCLIP
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        x = x @ self.text_projection

        return x


class OSVG(nn.Module):
    def __init__(self, args):
        super(OSVG, self).__init__()
        print("This is the OSVG model.")
        # CLIP Model name: ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        if (args.model == "ViT-B-16-dfn2b"):
            print("init ViT-B-16-dfn2b")
            self.clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='dfn2b', device=args.device)
            self.extract_layer = [0, 3, 7, 11]
            self.patch_size = 14
        elif (args.model == "ViT-B/32"):
            print("init ViT-B/32")
            self.clip, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=args.device)
            self.extract_layer = [0, 7, 15, 23]
            self.patch_size = 32
        else:  # default
            print("init ViT-B/16")
            self.clip, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=args.device)
            self.extract_layer = [0, 3, 7, 11]
            self.patch_size = 16
            
        # os
        self.text_resize = FeatureResizer(
            input_feat_size=512,
            output_feat_size=768,
            dropout=0.1,
        )
        # resize pos embed
        if self.clip.visual.image_size[0] != args.imsize:
            resize_pos_embed(self.clip.visual, self.patch_size, args.imsize)
        hidden_dim = self.clip.transformer.width
        self.image_encoder_clip_vg = MultiLevel_ImageEncoder_modified(self.clip.visual, self.extract_layer)
        self.text_encoder_clip_vg = TextEncoder_modified(self.clip)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.ml_visu_proj = nn.Linear(len(self.extract_layer) * self.clip.visual.transformer.width, hidden_dim)

    def tensorize_inputs(self, images: NestedTensor, texts: NestedTensor):
        image_tensors = images.tensors
        texts_tensors = texts.tensors
        return image_tensors, texts_tensors

    def forward(self, img_data, text_data):
        image_tensors, text_tensors = self.tensorize_inputs(img_data, text_data)
        text_features = self.text_encoder_clip_vg(text_tensors)  # B * 77 * 512
        image_features = self.image_encoder_clip_vg(image_tensors, self.text_resize(text_features))  # B * 197 * 512 
        cls_token = image_features[:, 0,:]
        visu_src = self.ml_visu_proj(cls_token)  # L B 4H -> L B H
        pred_box = self.bbox_embed(visu_src).sigmoid()
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