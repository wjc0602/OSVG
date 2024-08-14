import os
import random
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import eval_utils
from utils.box_utils import xywh2xyxy
import utils.misc as utils
from models.osvg_clip_eval import OSVG
from datasets import build_dataset
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-VG Args', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    # only support ViT-B/16 and ViT-L/14
    parser.add_argument('--model', type=str, default='ViT-B/16', help="Name of model to be exploited.")
    parser.add_argument('--ce', action='store_true', help="If true, use gaussian blur augmentation")
    parser.add_argument('--ce_keep', default=1.0, type=float, help='image size')
    parser.add_argument('--min_keep', default=0.0, type=float, help='image size')
    parser.add_argument('--max_keep', default=1.0, type=float, help='image size')
    parser.add_argument('--imsize', default=384, type=int, help='image size')
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='image_data', help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='split_data',  help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='unc', type=str, help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_query_len', default=77, type=int, help='maximum time steps (lang length) per batch')
    parser.add_argument('--prompt', type=str, default='{pseudo_query}', help="Prompt template")
    # dataset parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # evalutaion options
    parser.add_argument('--eval_set', default='testA', type=str)  # 'testA', 'testB', 'val'
    parser.add_argument('--eval_model', default='outputs/referit/osvg_clip_vitb16_384_g2_b64_c0.7/best_checkpoint.pth', type=str)

    return parser


def main(args):
    device = torch.device(args.device)
    # # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('### INFO ### torch.backends.cudnn.benchmark = {}'.format(torch.backends.cudnn.benchmark))

    # build model
    model = OSVG(args)
    model.to(device)

    n_parameters_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of requires_grad params: ', n_parameters_grad)
    print('number of all params: ', n_parameters)

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("Current model training epoch is: ", checkpoint['epoch'])

    # output log
    eval_model = args.eval_model
    eval_model = eval_model.split('/')[-1].split('.')[0]

    pred_box_list = []
    gt_box_list = []
    save_dir_vts = os.path.join('results_vts', args.dataset, args.eval_set)
    save_dir_box = os.path.join('results_box', args.dataset, args.eval_set)
    os.makedirs(save_dir_vts, exist_ok=True)
    os.makedirs(save_dir_box, exist_ok=True)
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader_test)):
            model.eval()
            img_data, text_data, target, image_name, txt, img_w, img_h = batch
            # copy to GPU
            if args.min_keep < target[0][2]*target[0][3] < args.max_keep:
                img_data = img_data.to(device)
                text_data = text_data.to(device)
                target = target.to(device)
                output, removed_indexes = model(img_data, text_data, args.ce_keep)
                pred_box_list.append(output.cpu())
                gt_box_list.append(target.cpu())

            # 复原图片
            input_array = img_data.tensors.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
            input_array = (input_array * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            input_uint8 = np.uint8(input_array * 255)
            img = cv2.cvtColor(input_uint8, cv2.COLOR_RGB2BGR)

            # 绘制矩形框
            img_bbox= img.copy()
            bbox =xywh2xyxy(target).squeeze() * args.imsize
            cv2.rectangle(img_bbox, [int(bbox[0]),int(bbox[1])], [int(bbox[2]),int(bbox[3])], color=(0,0,255), thickness=2)
            save_path = os.path.join(save_dir_vts, f'{txt[0]}_{image_name[0]}')
            cv2.imwrite(save_path, img_bbox)

            # TokenSelect
            for j in range(len(removed_indexes)):  # for every select stage
                indexs = removed_indexes[j]
                for k in indexs[0]:
                    x = int(k.item() // 24)
                    y = int(k.item() % 24)
                    img[x * 16:(x + 1) * 16, y * 16:(y + 1) * 16, :] = 1
                save_path = os.path.join(save_dir_vts ,f"{txt[0]}_{image_name[0]}_{j + 1}.jpg")
                cv2.imwrite(save_path, img)

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)
    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    print(accuracy)
    print(f"num:{len(pred_box_list)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP-VG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
