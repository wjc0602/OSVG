import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import HungarianMatcherDynamicK, SetCriterionDynamicK
from .vl_transformer import build_vl_transformer
from .head import DynamicHead
from .clip import *
from torchvision.transforms import Resize
from utils.misc import NestedTensor
from detectron2.layers import batched_nms
from detectron2.structures import Boxes, Instances
from collections import namedtuple
from utils.box_utils import xywh2xyxy

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


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
        self.input_resolution = clip_visu_model.input_resolution
        self.output_dim = clip_visu_model.output_dim
        self.conv1 = clip_visu_model.conv1
        self.class_embedding = clip_visu_model.class_embedding
        self.positional_embedding = clip_visu_model.positional_embedding
        self.ln_pre = clip_visu_model.ln_pre
        self.transformer = MultiLevel_Transformer(clip_visu_model.transformer, extract_layer)
        self.ln_post = clip_visu_model.ln_post
        self.proj = clip_visu_model.proj
        self.positional_embedding.requires_grad_(True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                       dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND, B L H -> L B H
        ml_x = self.transformer(x)
        x = torch.cat(ml_x, dim=2)
        x = x.permute(1, 0, 2)  # LND -> NLD, L B H -> B 4*L H
        return x


class MutiLevel_TextEncoder_modified(nn.Module):
    def __init__(self, clip_model, extract_layer):
        super().__init__()
        self.transformer = MultiLevel_Transformer(clip_model.transformer, extract_layer)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, x):
        x = self.token_embedding(x).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        ml_x = self.transformer(x)
        x = torch.cat(ml_x, dim=2)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x


class ImageEncoder_modified(nn.Module):
    def __init__(self, clip_visu_model):
        super().__init__()
        self.input_resolution = clip_visu_model.input_resolution
        self.output_dim = clip_visu_model.output_dim
        self.conv1 = clip_visu_model.conv1
        self.class_embedding = clip_visu_model.class_embedding
        self.positional_embedding = clip_visu_model.positional_embedding
        self.ln_pre = clip_visu_model.ln_pre
        self.transformer = clip_visu_model.transformer
        self.ln_post = clip_visu_model.ln_post
        self.proj = clip_visu_model.proj

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # TODO: ModifiedCLIP
        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj

        return x


class TextEncoder_modified(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
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

class Diff_VG(nn.Module):
    def __init__(self, args):
        super(Diff_VG, self).__init__()
        print("This is the Diff_VG model.")
        # CLIP Model name: ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        if (args.model == "ViT-L/14"):
            print("init ViT-L/14")
            self.clip, _ = clip.load("ViT-L/14", device=args.device)
            self.extract_layer = [0, 7, 15, 23]
            self.patch_size = 14
        elif (args.model == "ViT-B/32"):
            print("init ViT-B/32")
            self.clip, _ = clip.load("ViT-B/32", device=args.device)
            self.extract_layer = [0, 3, 7, 11]
            self.patch_size = 32
        else:  # default
            print("init ViT-B/16")
            self.clip, _ = clip.load("ViT-B/16", device=args.device)
            self.extract_layer = [0, 3, 7, 11]
            self.patch_size = 16

        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)

        hidden_dim = self.clip.transformer.width
        self.visu_proj = nn.Linear(512, hidden_dim)
        self.text_proj = nn.Linear(self.clip.transformer.width, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)
        self.imsize = args.imsize
        self.num_visu_token = int((self.imsize / self.patch_size) ** 2)
        self.num_text_token = args.max_query_len
        num_total = self.num_visu_token + 1 + self.num_text_token + 1  # VISU token + [cls] + TEXT token + [REG]token
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)

        self.vl_transformer = build_vl_transformer(args)

        self.image_encoder_clip_vg = MultiLevel_ImageEncoder_modified(self.clip.visual, self.extract_layer)
        self.text_encoder_clip_vg = TextEncoder_modified(self.clip)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.ml_visu_proj = nn.Linear(len(self.extract_layer) * self.clip.visual.transformer.width, hidden_dim)

        # Diffusion
        self.num_classes = 1
        self.num_proposals = 50
        self.num_heads = 6
        timesteps = 1000
        sampling_timesteps = 1
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = 2.0
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        self.head = DynamicHead(num_classes = self.num_classes, roi_input_shape=[[14,14],[14,14],[14,14],[14,14],])
        # Loss parameters:
        class_weight = 2.0
        giou_weight = 2.0
        l1_weight = 5.0
        no_object_weight = 0.1
        self.deep_supervision = True
        self.use_focal = True
        self.use_nms = True

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)
    
    def tensorize_inputs(self, images: NestedTensor, texts: NestedTensor):
        image_tensors = images.tensors
        texts_tensors = texts.tensors

        return image_tensors, texts_tensors

    def get_masks(self, images: NestedTensor, texts: NestedTensor):
        torch_resize = Resize([int(self.imsize / self.patch_size), int(self.imsize / self.patch_size)])
        visu_masks = torch_resize(images.mask)  # 14 * 14 = 196， or， 16 * 16 = 256
        visu_masks = visu_masks.to(torch.bool)
        visu_masks = visu_masks.flatten(1)  # visu_mask：B*L, torch.Size([B, 196])
        text_masks = texts.mask.to(torch.bool)
        text_masks = ~text_masks
        text_masks = text_masks.flatten(1)  # text_mask：B*L, torch.Size([B, 77])
        assert text_masks is not None

        return visu_masks, text_masks

    def forward(self, img_data, text_data, gt_instances, training):
        # gt_instances  xywh
        batch_size = img_data.tensors.shape[0]
        image_tensors, text_tensors = self.tensorize_inputs(img_data, text_data)
        image_features = self.image_encoder_clip_vg(image_tensors.type(self.clip.dtype))  # B * 197 * 512
        text_features = self.text_encoder_clip_vg(text_tensors)  # B * 77 * 512
        visu_src = self.ml_visu_proj(image_features.float())  # L B 4H -> L B H
        text_src = self.text_proj(text_features.float())  # B * 77 * 512
        # permute BxLenxC to LenxBxC
        visu_src = visu_src.permute(1, 0, 2)  # 197 * 4 * 512
        text_src = text_src.permute(1, 0, 2)  # 77 * 4 * 512
        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, batch_size, 1)  # 1 * B * hidden_dim
        # (1 + 77 + 197) * B * 512 = 275 * B * 512; VIT-L/14: (1 + 77 + 257) * B * 768
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        visu_mask, text_mask = self.get_masks(img_data, text_data)
        tgt_mask = torch.zeros((batch_size, 1)).to(tgt_src.device).to(torch.bool)
        cls_mask = torch.zeros((batch_size, 1)).to(tgt_src.device).to(torch.bool)
        vl_mask = torch.cat([tgt_mask, text_mask, cls_mask, visu_mask], dim=1)
        # (1 + 77 + 1 + 196) * B * H = 275 * B * 512
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        features = vg_hs[1:197,:,:].permute(1, 2, 0).reshape(batch_size, 512, 14, 14)

        # Diffusion text
        if not training:
            results = self.ddim_sample(gt_instances, features) # gt_instances cxcywh   features 8,512,14,14
            return results
        
        # Diffusion train
        if training:
            targets, x_boxes, noises, t = self.prepare_targets(gt_instances) # gt_instances cxcywh -> x_boxes xyxy
            # x_boxes: xyxy
            t = t.squeeze(-1)
            x_boxes = x_boxes * self.imsize
            outputs_class, outputs_coord = self.head(features, x_boxes, t, None) # xyxy
            # outputs_class min -8.4329 max -0.2925 mean -4.5660
            # outputs_coord 
            # [6, 8, 50, 2] [6, 8, 50, 4] [nhead, bs, nbox, ...]
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]} # 最后一个head的输出
            # pred_logits[6, 32, 300, 80] outputs_coord[6, 32, 300, 4]
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
    

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    

    def model_predictions(self, backbone_feats, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * self.imsize
        outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / self.imsize
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale) # max 2  min -2
        pred_noise = self.predict_noise_from_start(x, t, x_start) # max 3.4888   min -4.0882

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, clip_denoised=True, do_postprocess=True):
        batch = backbone_feats.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=batched_inputs.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=batched_inputs.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, img, time_cond, self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            # x_start (-2,2)  pred_noise (-4.0882, 3.4888)
            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image) #[0.35, 0.2244 ]
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(224)
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            # outputs_coord 基于224的box
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred)
        results /= self.imsize
        return results

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def prepare_diffusion_repeat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=gt_boxes.device).long()
        noise = torch.randn(self.num_proposals, 4, device=gt_boxes.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=gt_boxes.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=gt_boxes.device)

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=gt_boxes.device).long()
        noise = torch.randn(self.num_proposals, 4, device=gt_boxes.device)
        # noise sample
        box_placeholder = torch.randn(self.num_proposals - 1, 4, device=gt_boxes.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
        box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
        x_start = torch.cat((gt_boxes, box_placeholder), dim=0)

        x_start = (x_start * 2. - 1.) * self.scale
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)
        # min: -0.3776   max: 1.3564
        return diff_boxes, noise, t

    def prepare_targets(self, gt_boxes):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in gt_boxes: # [cx,cy,w,h]
            target = {}
            image_size_xyxy = torch.as_tensor([224, 224, 224, 224], dtype=torch.float, device=gt_boxes.device)
            gt_boxes = targets_per_image.unsqueeze(0) # [1, 4] [cx,cy,w,h]
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes) # cxcywh -> xyxy
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = torch.zeros(1, dtype=int).to(gt_boxes.device)
            target["boxes"] = gt_boxes # norm cxcywh
            target["boxes_xyxy"] = xywh2xyxy(gt_boxes * image_size_xyxy) # denorm xyxy
            target["image_size_xyxy"] = image_size_xyxy.to(gt_boxes.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(gt_boxes.device)
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)
    
    def inference(self, box_cls, box_pred):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(box_pred)
        results = []
        results_boxes = []
        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=box_cls.device).unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image) in enumerate(zip(scores, box_pred)):
                result = Instances(224)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(1, sorted=False) # 获取分数最高的一个
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                    
                results_boxes.append(box_pred_per_image) # diffvg
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)
        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image) in enumerate(zip(scores, labels, box_pred)):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(224)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)
        
        return torch.cat(results_boxes,dim=0)
        # return results



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


