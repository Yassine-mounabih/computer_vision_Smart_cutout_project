"""Tests supplémentaires : mono-clic, entrée/sortie, précautions de fusion."""
from pathlib import Path
import sys
import numpy as np
import cv2
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.generate_demo import make_sample
from segmentation.one_click import propose_click_rect, segment_one_click, keep_clicked_component
from segmentation.grabcut import transparent_cutout
from segmentation.metrics import mask_metrics
from segmentation.refine import load_model, refine_with_unet

@pytest.mark.parametrize('seed',[900,901,902,903,904,905])
def test_one_click_six_demo_objects(seed):
    image,reference=make_sample(seed,160)
    yy,xx=np.where(reference>0)
    point=(int(np.median(xx)),int(np.median(yy)))
    result,rect=segment_one_click(image,point,iterations=3)
    assert result.shape==reference.shape
    assert mask_metrics(result,reference)['IoU']>.8
    x,y,w,h=rect
    assert 0<x<x+w<image.shape[1]
    assert 0<y<y+h<image.shape[0]


def test_clicked_component_only():
    mask=np.zeros((60,80),dtype=np.uint8)
    mask[10:25,10:25]=1;mask[30:45,55:70]=1
    result=keep_clicked_component(mask,(15,15))
    assert result.sum()==15*15
    assert result[35,60]==0


def test_click_outside_image_rejected():
    im=np.zeros((80,80,3),dtype=np.uint8)
    with pytest.raises(ValueError):propose_click_rect(im,(-10,15))


def test_rgb_is_not_repainted():
    rng=np.random.default_rng(78)
    image=rng.integers(0,256,(23,29,3),dtype=np.uint8)
    mask=(rng.random((23,29))>.5).astype(np.uint8)
    rgba=transparent_cutout(image,mask)
    assert np.array_equal(image,rgba[...,:3])
    assert np.array_equal(mask*255,rgba[...,3])


def test_checkpoint_and_fusion_bounds():
    root=Path(__file__).resolve().parents[1]
    model,_=load_model(root/'models/unet_demo.pt')
    image,gt=make_sample(918,128)
    coarse=cv2.erode(gt,np.ones((5,5),np.uint8))
    refined,probs=refine_with_unet(image,coarse,model)
    assert refined.shape==coarse.shape==probs.shape
    assert np.isfinite(probs).all()
    assert probs.min()>=0 and probs.max()<=1
    changes=int((refined!=coarse).sum())
    assert changes<=int(.12*coarse.sum())+1
