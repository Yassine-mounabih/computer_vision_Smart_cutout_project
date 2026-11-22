import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.generate_demo import make_sample
from scripts.train_unet import rect_from_mask
from segmentation.grabcut import segment_grabcut,transparent_cutout,normalize_rect
from segmentation.metrics import mask_metrics
from segmentation.refine import prepare_tensor
from segmentation.unet import UNetRefiner


def test_grabcut():
    image,gt=make_sample(903,128)
    mask=segment_grabcut(image,rect_from_mask(gt,9),iterations=3)
    assert mask.shape==gt.shape
    assert mask.dtype==np.uint8
    assert 0<mask.sum()<mask.size


def test_transparency():
    image=np.ones((10,10,3),dtype=np.uint8)*125
    mask=np.zeros((10,10),dtype=np.uint8);mask[2:5,2:5]=1
    result=transparent_cutout(image,mask)
    assert result.shape==(10,10,4)
    assert result[3,3,3]==255 and result[0,0,3]==0


def test_metrics():
    a=np.array([[1,0],[1,0]],dtype=np.uint8)
    b=np.array([[1,1],[0,0]],dtype=np.uint8)
    scores=mask_metrics(a,b)
    assert scores['IoU']==pytest.approx(1/3)
    assert scores['Dice']==pytest.approx(1/2)


def test_unet_shapes():
    import torch
    torch.set_num_threads(2)
    model=UNetRefiner(base_channels=8).eval()
    img=np.zeros((64,64,3),dtype=np.uint8)
    mask=np.ones((64,64),dtype=np.uint8)
    x=prepare_tensor(img,mask)
    with torch.inference_mode():
        out=model(x)
    assert out.shape==(1,1,128,128)


def test_rect_normalization():
    assert normalize_rect((30,40,-20,-30),100,100)==(10,10,20,30)
    with pytest.raises(ValueError):normalize_rect((0,0,0,0),100,100)
