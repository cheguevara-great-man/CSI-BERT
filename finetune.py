from model import CSIBERT, Token_Classifier
from transformers import BertConfig
import argparse
import tqdm
import torch
from dataset import Widar_digit_amp_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import numpy as np

pad = -1000


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x = x + identity
        return self.relu(x)


class WidarDigit_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        i_downsample = None
        layers = []
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            i_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )
        layers.append(ResBlock(self.in_channels, planes, i_downsample=i_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion
        for _ in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))
        return nn.Sequential(*layers)


def WidarDigit_ResNet18(num_classes: int = 10):
    return WidarDigit_ResNet(Block, [2, 2, 2, 2], num_classes=num_classes)


class Recon_Classifier_Pipeline(nn.Module):
    def __init__(self, reconstructor, classifier, carrier_dim=90):
        super().__init__()
        self.reconstructor = reconstructor
        self.classifier = classifier
        self.carrier_dim = carrier_dim

    def forward(self, input_ids, attn_mask, mask_keep, timestamp):
        with torch.no_grad():
            x_pred = self.reconstructor(input_ids, attn_mask, timestamp)

        if mask_keep.dim() == 2:
            mask_keep = mask_keep.unsqueeze(2).repeat(1, 1, self.carrier_dim)

        x_rec = input_ids * mask_keep + x_pred * (1 - mask_keep)
        logits = self.classifier(x_rec)
        return logits


def _print_label_stats(name, dataset, class_num):
    if not hasattr(dataset, "items"):
        print(f"{name} label stats: unavailable")
        return
    labels = np.array([item.get("label", -1) for item in dataset.items], dtype=np.int64)
    if labels.size == 0:
        print(f"{name} label stats: empty")
        return
    unique, counts = np.unique(labels, return_counts=True)
    pairs = ", ".join([f"{int(k)}:{int(v)}" for k, v in zip(unique, counts)])
    print(f"{name} label min/max: {int(labels.min())}/{int(labels.max())}")
    print(f"{name} label counts: {pairs}")
    if labels.min() < 0 or labels.max() >= class_num:
        print(f"{name} label out of range for class_num={class_num}")

def _check_label_alignment(name, dataset, max_samples=200):
    if not hasattr(dataset, "items") or not hasattr(dataset, "_load_shard"):
        print(f"{name} label align: unavailable")
        return
    total = min(max_samples, len(dataset.items))
    if total == 0:
        print(f"{name} label align: empty")
        return
    mismatch0 = 0
    mismatch1 = 0
    out0 = 0
    out1 = 0
    for i in range(total):
        row = dataset.items[i]
        shard = dataset._load_shard(row["shard_id"])
        y = shard.get("y", None)
        if y is None:
            print(f"{name} label align: shard has no 'y'")
            return
        off = int(row["offset"])
        label = int(row["label"])
        if 0 <= off < y.shape[0]:
            if int(np.asarray(y[off]).reshape(-1)[0]) != label:
                mismatch0 += 1
        else:
            out0 += 1
        off1 = off - 1
        if 0 <= off1 < y.shape[0]:
            if int(np.asarray(y[off1]).reshape(-1)[0]) != label:
                mismatch1 += 1
        else:
            out1 += 1
    print(f"{name} label align (offset): mismatch {mismatch0}/{total}, out_of_range {out0}")
    print(f"{name} label align (offset-1): mismatch {mismatch1}/{total}, out_of_range {out1}")

def _print_label_order_stats(name, dataset, batch_size, max_batches=5, head=50):
    if not hasattr(dataset, "items"):
        print(f"{name} label order: unavailable")
        return
    labels = [int(item.get("label", -1)) for item in dataset.items]
    if not labels:
        print(f"{name} label order: empty")
        return
    runs = []
    cur = labels[0]
    run_len = 1
    for v in labels[1:]:
        if v == cur:
            run_len += 1
        else:
            runs.append(run_len)
            cur = v
            run_len = 1
    runs.append(run_len)
    max_run = max(runs)
    avg_run = float(sum(runs)) / max(1, len(runs))
    print(f"{name} label order: max run {max_run}, avg run {avg_run:.2f}")
    show = labels[:head]
    print(f"{name} label head({len(show)}): {show}")
    total_batches = min(max_batches, (len(labels) + batch_size - 1) // batch_size)
    for i in range(total_batches):
        chunk = labels[i * batch_size:(i + 1) * batch_size]
        uniq = len(set(chunk))
        print(f"{name} batch {i} unique labels: {uniq}")

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mask_percent', type=float, default=0.15)
    parser.add_argument('--normal', action="store_true", default=False)
    parser.add_argument('--hs', type=int, default=64)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--intermediate_size', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--position_embedding_type', type=str, default="absolute")
    parser.add_argument('--time_embedding', action="store_true", default=False) # whether to use time embedding
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument("--carrier_dim", type=int, default=52)
    parser.add_argument("--carrier_attn", action="store_true",default=False)
    parser.add_argument("--freeze", action="store_true",default=False)
    parser.add_argument('--lr', type=float, default=0.0005)
    # parser.add_argument("--test_people", type=int, nargs='+', default=[0,1])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--class_num', type=int, default=6) #action:6, people:8
    parser.add_argument('--task', type=str, default="action") # "action" or "people"
    parser.add_argument("--path", type=str, default='./pretrain.pth')
    parser.add_argument("--no_pretrain", action="store_true",default=False)
    parser.add_argument('--data_path', type=str, default="/home/cxy/data/code/datasets/sense-fi/Widar_digit")
    parser.add_argument('--sample_rate', type=float, default=0.2)
    parser.add_argument('--sample_method', type=str, default="equidistant")
    parser.add_argument('--interpolation_method', type=str, default="linear")
    parser.add_argument('--use_mask_0', type=int, default=1)
    parser.add_argument('--use_x_gt', action="store_true", default=False)
    parser.add_argument('--print_x_gt', action="store_true", default=False)
    parser.add_argument('--label_from_shard', action="store_true", default=False)
    parser.add_argument('--offset_shift', type=int, default=0)
    parser.add_argument('--check_label_align', action="store_true", default=False)
    parser.add_argument('--label_check_samples', type=int, default=200)
    parser.add_argument('--check_label_order', action="store_true", default=False)
    parser.add_argument('--label_order_batches', type=int, default=5)
    parser.add_argument('--label_order_head', type=int, default=50)
    args = parser.parse_args()
    return args

def main():
    ACC=[]

    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    bertconfig=BertConfig(max_position_embeddings=args.max_len, hidden_size=args.hs, position_embedding_type=args.position_embedding_type,num_hidden_layers=args.layers,num_attention_heads=args.heads, intermediate_size=args.intermediate_size)
    csibert = CSIBERT(bertconfig, args.carrier_dim, args.carrier_attn, args.time_embedding)
    reconstructor = Token_Classifier(csibert, args.carrier_dim)
    if not args.no_pretrain:
        reconstructor.load_state_dict(torch.load(args.path, map_location=device))
    for param in reconstructor.parameters():
        param.requires_grad = False
    reconstructor.eval()

    classifier = WidarDigit_ResNet18(num_classes=args.class_num)
    model = Recon_Classifier_Pipeline(reconstructor, classifier, carrier_dim=args.carrier_dim)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=0.01)
    label_source = "shard" if args.label_from_shard else "index"
    train_data = Widar_digit_amp_dataset(
        root_dir=args.data_path,
        split="train",
        sample_rate=args.sample_rate,
        sample_method=args.sample_method,
        interpolation_method=args.interpolation_method,
        use_mask_0=args.use_mask_0,
        is_rec=1,
        label_source=label_source,
        offset_shift=args.offset_shift,
    )
    test_data = Widar_digit_amp_dataset(
        root_dir=args.data_path,
        split="test",
        sample_rate=args.sample_rate,
        sample_method=args.sample_method,
        interpolation_method=args.interpolation_method,
        use_mask_0=args.use_mask_0,
        is_rec=1,
        label_source=label_source,
        offset_shift=args.offset_shift,
    )
    _print_label_stats("train", train_data, args.class_num)
    _print_label_stats("test", test_data, args.class_num)
    if args.check_label_align:
        _check_label_alignment("train", train_data, args.label_check_samples)
        _check_label_alignment("test", test_data, args.label_check_samples)
    if args.check_label_order:
        _print_label_order_stats("train", train_data, args.batch_size, args.label_order_batches, args.label_order_head)
        _print_label_order_stats("test", test_data, args.batch_size, args.label_order_batches, args.label_order_head)
    if args.print_x_gt:
        x_in, mask_in, label_in, x_gt, timestamp = train_data[0]
        print("x_gt shape:", tuple(x_gt.shape))
        print("x_gt stats: min {:.6f} max {:.6f} mean {:.6f}".format(
            float(x_gt.min()), float(x_gt.max()), float(x_gt.mean())
        ))
        print("x_gt[0, :10]:", x_gt[0, :10].tolist())
        print("label:", int(label_in))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    loss_func = nn.CrossEntropyLoss()
    best_acc=0
    best_epoch=0

    j=0
    while True:
        j+=1
        model.train()
        model.reconstructor.eval()
        torch.set_grad_enabled(True)
        loss_list=[]
        acc_list=[]
        pbar = tqdm.tqdm(train_loader, disable=False)
        for x_in, mask_in, label_in, x_gt, timestamp in pbar:
            label = label_in.to(device)
            if args.use_x_gt:
                x = x_gt.to(device)
                y = model.classifier(x)
                loss = loss_func(y, label)
                output = torch.argmax(y, dim=-1)
                acc = torch.sum(output == label) / label.shape[0]

                model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optim.step()

                loss_list.append(loss.item())
                acc_list.append(acc.item())
                continue

            x = x_in.to(device)
            timestamp = timestamp.to(device)
            input = x.clone()
            max_values, _ = torch.max(input, dim=-2, keepdim=True)
            input[input == pad] = -pad
            min_values, _ = torch.min(input, dim=-2, keepdim=True)
            input[input == -pad] = pad

            mask_keep = mask_in.to(device).float()
            non_pad = mask_keep
            if non_pad.dim() == 3:
                non_pad = non_pad[:, :, 0]
            avg = copy.deepcopy(input)
            avg[input == pad] = 0
            avg = torch.sum(avg, dim=-2, keepdim=True) / (torch.sum(non_pad.unsqueeze(-1), dim=-2, keepdim=True)+1e-8)
            std = (input - avg) ** 2
            std[input == pad] = 0
            std = torch.sum(std, dim=-2, keepdim=True) / (torch.sum(non_pad.unsqueeze(-1), dim=-2, keepdim=True)+1e-8)
            std = torch.sqrt(std)

            if args.normal:
                input = (input - avg) / (std+1e-5)

            batch_size,seq_len,carrier_num=input.shape
            attn_mask = torch.ones_like(non_pad)
            if args.normal:
                rand_word = torch.tensor(csibert.mask(batch_size, std=torch.tensor([1]).to(device), avg=torch.tensor([0]).to(device))).to(device)
            else:
                rand_word = torch.tensor(csibert.mask(batch_size, std=std.to(device), avg=avg.to(device))).to(device)
            loss_mask = 1.0 - non_pad
            loss_mask_full = loss_mask.unsqueeze(2).repeat(1, 1, carrier_num)
            input[loss_mask_full == 1] = rand_word[loss_mask_full == 1]
            input[x==pad]=rand_word[x==pad]
            y = model(input, attn_mask, mask_keep, timestamp)

            loss = loss_func(y,label)
            output = torch.argmax(y, dim=-1)
            acc=torch.sum(output==label)/batch_size

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
            optim.step()

            loss_list.append(loss.item())
            acc_list.append(acc.item())
        log="Epoch {} | Train Loss {:06f}, Train Acc {:06f}, ".format(j,np.mean(loss_list),np.mean(acc_list))
        print(log)
        with open(args.task+".txt", 'a') as file:
            file.write(log)

        model.eval()
        torch.set_grad_enabled(False)
        loss_list=[]
        acc_list=[]
        pbar = tqdm.tqdm(test_loader, disable=False)
        for x_in, mask_in, label_in, x_gt, timestamp in pbar:
            label = label_in.to(device)
            if args.use_x_gt:
                x = x_gt.to(device)
                y = model.classifier(x)
                loss = loss_func(y, label)
                output = torch.argmax(y, dim=-1)
                acc = torch.sum(output == label) / label.shape[0]

                loss_list.append(loss.item())
                acc_list.append(acc.item())
                continue

            x = x_in.to(device)
            timestamp = timestamp.to(device)
            input = x.clone()
            max_values, _ = torch.max(input, dim=-2, keepdim=True)
            input[input == pad] = -pad
            min_values, _ = torch.min(input, dim=-2, keepdim=True)
            input[input == -pad] = pad

            mask_keep = mask_in.to(device).float()
            non_pad = mask_keep
            if non_pad.dim() == 3:
                non_pad = non_pad[:, :, 0]
            avg = copy.deepcopy(input)
            avg[input == pad] = 0
            avg = torch.sum(avg, dim=-2, keepdim=True) / (torch.sum(non_pad.unsqueeze(-1), dim=-2, keepdim=True)+1e-8)
            std = (input - avg) ** 2
            std[input == pad] = 0
            std = torch.sum(std, dim=-2, keepdim=True) / (torch.sum(non_pad.unsqueeze(-1), dim=-2, keepdim=True)+1e-8)
            std = torch.sqrt(std)

            if args.normal:
                input = (input - avg) / (std+1e-5)

            batch_size,seq_len,carrier_num=input.shape
            attn_mask = torch.ones_like(non_pad)
            if args.normal:
                rand_word = torch.tensor(csibert.mask(batch_size, std=torch.tensor([1]).to(device), avg=torch.tensor([0]).to(device))).to(device)
            else:
                rand_word = torch.tensor(csibert.mask(batch_size, std=std.to(device), avg=avg.to(device))).to(device)
            loss_mask = 1.0 - non_pad
            loss_mask_full = loss_mask.unsqueeze(2).repeat(1, 1, carrier_num)
            input[loss_mask_full == 1] = rand_word[loss_mask_full == 1]
            input[x==pad]=rand_word[x==pad]
            y = model(input, attn_mask, mask_keep, timestamp)

            loss = loss_func(y,label)
            output = torch.argmax(y, dim=-1)
            acc=torch.sum(output==label)/batch_size

            loss_list.append(loss.item())
            acc_list.append(acc.item())
        log="Test Loss {:06f}, Test Acc {:06f}".format(np.mean(loss_list),np.mean(acc_list))
        print(log)
        ACC.append(np.mean(acc_list))
        with open(args.task+".txt", 'a') as file:
            file.write(log+"\n")
        if np.mean(acc_list)>=best_acc:
            best_acc=np.mean(acc_list)
            torch.save(model.state_dict(), args.task+".pth")
            best_epoch=0
        else:
            best_epoch+=1
        if best_epoch>=args.epoch:
            break

    print("Acc Max:",np.max(ACC))
    print("Acc Mean:",np.max(ACC[-30:]))
    print("Acc Std:",np.max(ACC[-30:]))


if __name__ == '__main__':
    main()
