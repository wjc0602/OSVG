import time
import random
import argparse
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import eval_utils
import utils.misc as utils
from models.osvg_clip import OSVG
from datasets import build_dataset


def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, image_name, txt, img_w, img_h = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        output = model(img_data, text_data, args.ce_keep)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    torch.cuda.synchronize()
    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    return accuracy

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-VG Args', add_help=False)
    parser.add_argument('--sup_type', default='full', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    # only support ViT-B/16 and ViT-L/14
    parser.add_argument('--model', type=str, default='ViT-L/14', help="Name of model to be exploited.")
    parser.add_argument('--ce', action='store_true', help="If true, use gaussian blur augmentation")
    parser.add_argument('--ce_keep', default=1.0, type=float, help='image size')
    parser.add_argument('--ce_start', default=20, type=int, help='image size')
    parser.add_argument('--ce_warm', default=30, type=int, help='image size')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--imsize', default=384, type=int, help='image size')
    """ embedding size"""
    parser.add_argument('--emb_size', default=512, type=int, help='fusion module embedding dimensions')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='image_data', help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='split_data',  help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='unc', type=str, help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_query_len', default=50, type=int, help='maximum time steps (lang length) per batch')
    parser.add_argument('--prompt', type=str, default='{pseudo_query}', help="Prompt template")
    # dataset parameters
    parser.add_argument('--device', default='cpu', help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--retrain', default='', help='retrain from checkpoint')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # evalutaion options
    parser.add_argument('--eval_set', default='val', type=str)  # 'testA', 'testB', 'val'
    parser.add_argument('--eval_model', default='outputs/osvg_clip_vitb16_256_g2_b64_b5_g2_unc/best_checkpoint.pth', type=str)

    return parser


def main(args):
    """ distribution init """
    print("git:\n  {}\n".format(utils.get_sha()))
    if (args.model == "ViT-L/14" or args.model == "ViT-L/14@336px"):
        args.vl_hidden_dim = 768
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

    model_without_ddp = model
    n_parameters_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of requires_grad params: ', n_parameters_grad)
    print('number of all params: ', n_parameters)

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    # model_without_ddp.load_state_dict(checkpoint['model'])
    print("Current model training epoch is: ", checkpoint['epoch'])

    # output log
    eval_model = args.eval_model
    eval_model = eval_model.split('/')[-1].split('.')[0]
    start_time = time.time()

    # perform evaluation
    with torch.no_grad():
        accuracy = evaluate(args, model, data_loader_test, device)

    if utils.is_main_process():
        total_time = time.time() - start_time
        print('Testing time {}'.format(total_time))
        log_stats = {'test_model:': args.eval_model,'%s_set_accuracy'%args.eval_set: accuracy,}
        print(log_stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP-VG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
