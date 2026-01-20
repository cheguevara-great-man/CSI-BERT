from model import CSIBERT,Token_Classifier
from transformers import BertConfig
import argparse
import tqdm
import torch
from dataset import Widar_digit_amp_dataset
from torch.utils.data import DataLoader
import copy
import numpy as np

pad=np.array([-1000]*52)

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--normal', action="store_true", default=False) # whether to use norm layer
    parser.add_argument('--hs', type=int, default=64)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=100) # max input length
    parser.add_argument('--intermediate_size', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--position_embedding_type', type=str, default="absolute")
    parser.add_argument('--time_embedding', action="store_true", default=False) # whether to use time embedding
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument("--carrier_dim", type=int, default=52)
    parser.add_argument("--carrier_attn", action="store_true",default=False)
    parser.add_argument("--path", type=str, default='./pretrain.pth')
    parser.add_argument('--data_path', type=str, default="/home/cxy/data/code/datasets/sense-fi/Widar_digit")
    parser.add_argument('--sample_rate', type=float, default=0.2)
    parser.add_argument('--sample_method', type=str, default="equidistant")
    parser.add_argument('--interpolation_method', type=str, default="linear")
    parser.add_argument('--use_mask_0', type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device_name = "cuda:" + args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    bertconfig=BertConfig(max_position_embeddings=args.max_len, hidden_size=args.hs, position_embedding_type=args.position_embedding_type,num_hidden_layers=args.layers,num_attention_heads=args.heads, intermediate_size=args.intermediate_size)
    csibert = CSIBERT(bertconfig, args.carrier_dim, args.carrier_attn, args.time_embedding)
    csi_dim=args.carrier_dim
    model=Token_Classifier(csibert,csi_dim)
    model.load_state_dict(torch.load(args.path))
    model = model.to(device)
    data = Widar_digit_amp_dataset(
        root_dir=args.data_path,
        split="test",
        sample_rate=args.sample_rate,
        sample_method=args.sample_method,
        interpolation_method=args.interpolation_method,
        use_mask_0=args.use_mask_0,
        is_rec=1,
    )
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    model.eval()
    torch.set_grad_enabled(False)
    pbar = tqdm.tqdm(data_loader, disable=False)
    output1 = None
    output2 = None
    mse_list = []
    for x_in, mask_in, _, x_gt, timestamp in pbar:
        x = x_gt.to(device)
        timestamp = timestamp.to(device)
        attn_mask = (x[:, :, 0] != pad[0]).float().to(device)
        input = x_in.to(device).clone()
        batch_size, seq_len, carrier_num = input.shape
        max_values, _ = torch.max(input, dim=-2, keepdim=True)
        input[input == pad[0]] = -pad[0]
        min_values, _ = torch.min(input, dim=-2, keepdim=True)
        input[input == -pad[0]] = pad[0]
        non_pad = mask_in.to(device).float()
        if non_pad.dim() == 3:
            non_pad = non_pad[:, :, 0]
        avg = copy.deepcopy(input)
        avg[input == pad[0]] = 0
        avg = torch.sum(avg, dim=-2, keepdim=True) / (torch.sum(non_pad, dim=-2, keepdim=True) + 1e-8)
        std = (input - avg) ** 2
        std[input == pad[0]] = 0
        std = torch.sum(std, dim=-2, keepdim=True) / (torch.sum(non_pad, dim=-2, keepdim=True) + 1e-8)
        std = torch.sqrt(std)
        if args.normal:
            input = (input - avg) / (std + 1e-8)

        if args.normal:
            rand_word = torch.tensor(csibert.mask(batch_size, std=torch.tensor([1]).to(device), avg=torch.tensor([0]).to(device))).to(device)
        else:
            rand_word = torch.tensor(csibert.mask(batch_size, std=std.to(device), avg=avg.to(device))).to(device)
        loss_mask = 1.0 - non_pad
        loss_mask_full = loss_mask.unsqueeze(2).repeat(1, 1, carrier_num)
        input[loss_mask_full == 1] = rand_word[loss_mask_full == 1]
        input[x==pad[0]]=rand_word[x==pad[0]]
        if args.time_embedding:
            y = model(input, None, timestamp)
        else:
            y = model(input, None)
        if args.normal:
            y = y * std + avg

        non_pad_full = non_pad.unsqueeze(2).repeat(1, 1, carrier_num)
        output = input * non_pad_full + y * (1 - non_pad_full)
        mse_list.append(torch.mean((output - x) ** 2).item())

        if output1 is None:
            output1=y
            output2=output
        else:
            output1=torch.cat([output1,y],dim=0)
            output2=torch.cat([output2,output],dim=0)

    replace=output1.cpu().numpy()
    recover=output2.cpu().numpy()
    np.save("replace.npy",replace)
    np.save("recover.npy", recover)
    if len(mse_list) > 0:
        print("Recover MSE:", float(np.mean(mse_list)))

if __name__ == '__main__':
    main()
