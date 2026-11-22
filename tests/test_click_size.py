"""Le contrôle de taille du clic rapide change réellement la boîte GrabCut."""
import numpy as np
import pytest
from scripts.generate_demo import make_sample
from segmentation.one_click import propose_click_rect, segment_one_click


def test_scaling_click_zone_changes_proposal():
    image, mask = make_sample(900, 160)
    ys,xs=np.where(mask>0)
    point=(int(np.median(xs)),int(np.median(ys)))
    small=propose_click_rect(image, point, zone_scale=.5)
    default=propose_click_rect(image, point)
    large=propose_click_rect(image, point, zone_scale=1.5)
    area=lambda r: r[2]*r[3]
    assert area(small)<area(default)<area(large)
    for r in [small,default,large]:
        x,y,w,h=r
        assert x<=point[0]<x+w and y<=point[1]<y+h
        assert 0<x and 0<y and x+w<image.shape[1] and y+h<image.shape[0]


def test_one_click_accepts_size_parameter():
    image, mask = make_sample(901,160)
    ys,xs=np.where(mask>0)
    click=(int(np.median(xs)),int(np.median(ys)))
    result,rect=segment_one_click(image,click,iterations=2,zone_scale=1.35)
    assert result.shape==mask.shape
    assert rect == propose_click_rect(image,click,zone_scale=1.35)


@pytest.mark.parametrize('bad',[.3,2.2,float('nan')])
def test_bad_zone_scale_rejected(bad):
    im=np.zeros((80,80,3),dtype=np.uint8)
    with pytest.raises(ValueError,match='taille de la zone clic'):
        propose_click_rect(im,(40,40),zone_scale=bad)
