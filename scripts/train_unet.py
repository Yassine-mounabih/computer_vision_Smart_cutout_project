"""Entraîner U-Net pour corriger un masque GrabCut (images + vrais masques).

Par défaut : exemples synthétiques, suffisants pour une démo pédagogique.
Pour une utilisation réelle, utiliser --images et --masks avec photos annotées.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.generate_demo import make_sample, make_mixed_sample
from segmentation.grabcut import segment_grabcut
from segmentation.refine import MODEL_SIZE
from segmentation.unet import UNetRefiner


def rect_from_mask(mask: np.ndarray, pad: int = 12) -> tuple[int,int,int,int]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return 2,2, mask.shape[1]-4,mask.shape[0]-4
    return (max(1,int(xs.min())-pad), max(1,int(ys.min())-pad),
            min(mask.shape[1]-2,int(xs.max())+pad)-max(1,int(xs.min())-pad),
            min(mask.shape[0]-2,int(ys.max())+pad)-max(1,int(ys.min())-pad))


def coarse_from_gt_bbox(image: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """La vérité terrain sert uniquement à former une boîte durant la préparation."""
    rect = rect_from_mask(gt, max(7, int(image.shape[0]*0.08)))
    try:
        return segment_grabcut(image, rect, iterations=3)
    except (ValueError, cv2.error):
        return np.zeros_like(gt)


def load_pairs(images_dir: Path, masks_dir: Path, size: int = MODEL_SIZE):
    pairs = []
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'})
    for path in image_paths:
        mask_path=next((masks_dir/(path.stem+ext) for ext in ['.png','.jpg','.jpeg','.bmp']
                        if (masks_dir/(path.stem+ext)).exists()),None)
        if mask_path is None:
            continue
        img=np.array(Image.open(path).convert('RGB').resize((size,size)))
        gt=np.array(Image.open(mask_path).convert('L').resize((size,size),Image.Resampling.NEAREST))>127
        if gt.sum() < 25:
            continue
        pairs.append((img,gt.astype(np.uint8)))
    if len(pairs)<4:
        raise ValueError('Il faut au moins 4 paires image/masque ayant le même nom.')
    return pairs


class PairDataset(Dataset):
    def __init__(self,pairs,augment: bool):
        self.samples=[]
        for image,gt in pairs:
            coarse = coarse_from_gt_bbox(image,gt)
            self.samples.append((image,gt,coarse))
        self.augment=augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        image,gt,coarse = self.samples[index]
        image, gt, coarse = image.copy(),gt.copy(),coarse.copy()
        if self.augment:
            if random.random()<0.5:
                image=np.ascontiguousarray(image[:,::-1]);gt=np.ascontiguousarray(gt[:,::-1]);coarse=np.ascontiguousarray(coarse[:,::-1])
            # Petites erreurs au bord du masque, similaires à celles de GrabCut.
            if random.random()<0.65:
                k=random.choice([3,5,7,9])
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
                if random.random()<0.5:
                    coarse=cv2.erode(coarse,kernel,iterations=1)
                else:
                    coarse=cv2.dilate(coarse,kernel,iterations=1)
            if random.random()<0.3:
                image=np.uint8(np.clip(image.astype(np.float32)*random.uniform(0.85,1.15),0,255))
        inputs=np.concatenate((image.astype(np.float32)/255,coarse[...,None].astype(np.float32)),axis=2)
        return (torch.from_numpy(np.ascontiguousarray(inputs.transpose(2,0,1))),
                torch.from_numpy(np.ascontiguousarray(gt[None].astype(np.float32))))


def dice_loss(logits, target):
    p=torch.sigmoid(logits)
    inter=(p*target).sum(dim=(2,3))
    return (1-(2*inter+1)/(p.sum((2,3))+target.sum((2,3))+1)).mean()


def train(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    if args.images or args.masks:
        if not args.images or not args.masks:
            raise ValueError('Fournissez à la fois --images et --masks.')
        pairs=load_pairs(args.images,args.masks)
        random.shuffle(pairs)
        k=max(1,int(len(pairs)*0.2))
        train_pairs,val_pairs=pairs[k:],pairs[:k]
        source='photos annotées fournies par l’utilisateur'
    else:
        generator=make_mixed_sample if args.mixed_backgrounds else make_sample
        train_pairs=[generator(i,MODEL_SIZE) for i in range(args.num_demo)]
        val_pairs=[generator(700+i,MODEL_SIZE) for i in range(max(12,args.num_demo//5))]
        source=('composites synthétiques avec fonds photographiques CC0 — '
                'pas de véritables objets photographiques annotés'
                if args.mixed_backgrounds else
                'images synthétiques (démonstration pédagogique)')
    train_ds=PairDataset(train_pairs,True);val_ds=PairDataset(val_pairs,False)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False)
    device='cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    model=UNetRefiner(base_channels=args.base_channels).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    bce=nn.BCEWithLogitsLoss()
    best=float('inf')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(args.epochs):
        model.train();train_loss=0
        for x,y in train_loader:
            x=x.to(device);y=y.to(device)
            pred=model(x);loss=0.6*bce(pred,y)+0.4*dice_loss(pred,y)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            train_loss+=loss.item()*x.shape[0]
        model.eval();val_loss=0
        with torch.inference_mode():
            for x,y in val_loader:
                x=x.to(device);y=y.to(device)
                pred=model(x);loss=0.6*bce(pred,y)+0.4*dice_loss(pred,y)
                val_loss+=loss.item()*x.shape[0]
        train_loss/=len(train_ds);val_loss/=len(val_ds)
        print(f'Époque {epoch+1:02d}/{args.epochs:02d}: perte apprentissage={train_loss:.4f}, validation={val_loss:.4f}',flush=True)
        if val_loss<best:
            best=val_loss
            torch.save({'state_dict':model.state_dict(),'base_channels':args.base_channels,
                        'training_source':source,'val_loss':best,'epoch':epoch+1,'size':MODEL_SIZE},args.output)
    print(f'OK : modèle enregistré : {args.output}\nSource : {source}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Entraîner le mini U-Net à partir de masques annotés.')
    p.add_argument('--images',type=Path)
    p.add_argument('--masks',type=Path)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=12)
    p.add_argument('--lr',type=float,default=0.002)
    p.add_argument('--num-demo',type=int,default=180)
    p.add_argument('--base-channels',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cuda',action='store_true')
    p.add_argument('--mixed-backgrounds',action='store_true',help='Utiliser des fonds de photographies variées pour les composites synthétiques.')
    p.add_argument('--output',type=Path,default=ROOT/'models/unet_demo.pt')
    train(p.parse_args())
