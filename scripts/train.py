#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from src.utils.config import load_yaml, merge_dicts
from src.data.registry import build_dataloader
from src.models.factory import build_model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--method-config', default=None)
    ap.add_argument('--task-config', default=None)
    ap.add_argument('--dataset-config', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args=ap.parse_args()

    cfg=load_yaml(args.config)
    for p in [args.method_config,args.task_config,args.dataset_config]:
        if p: cfg=merge_dicts(cfg, load_yaml(p))

    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model(cfg).to(device)
    train_loader=build_dataloader(cfg, split='train')

    print(f"[train] device={device} batches={len(train_loader)}")
    if args.dry_run:
        x,y=next(iter(train_loader))
        with torch.no_grad():
            out=model(x.to(device))
        print('[dry-run] x',tuple(x.shape),'y',tuple(y.shape),'out',tuple(out.shape))
        return

    # TODO: full training loop w/ methods (finetune/replay/distill/ewc)
    print('TODO: implement full training loop')

if __name__=='__main__':
    main()
