import numpy as np
import pytest
from segmentation.pretrained_u2netp import refine_with_pretrained,ensure_weights

class Dummy:
    def __init__(self,prob): self.prob=prob
    def predict(self,img):return self.prob

def test_pretrained_refine_shape_and_no_changes_far_from_coarse():
    rgb=np.zeros((128,128,3),dtype=np.uint8)
    coarse=np.zeros((128,128),dtype=np.uint8)
    coarse[30:80,40:80]=1
    prob=np.zeros_like(coarse,dtype=np.float32)
    prob[26:84,35:85]=0.95
    prob[:10,:10]=0.95
    mask,p=refine_with_pretrained(rgb,coarse,Dummy(prob))
    assert mask.shape==coarse.shape and p.shape==coarse.shape
    assert mask[5,5]==0
    assert mask[35,45]==1
    assert mask[28,50]==1

def test_no_coarse_returns_empty():
    rgb=np.zeros((48,48,3),dtype=np.uint8)
    empty=np.zeros((48,48),dtype=np.uint8)
    result,p=refine_with_pretrained(rgb,empty,Dummy(np.ones_like(empty,dtype=np.float32)))
    assert result.sum()==0

def test_missing_model_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):ensure_weights(tmp_path/'not_found.onnx',download=False)
