"""Test GUI automatisé (exécuter sous un bureau graphique ou xvfb-run)."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from app import SegmentationApp
from segmentation.metrics import mask_metrics


def wait(root,app,seconds=25):
    t=time.monotonic()
    while app.busy or not app.task_queue.empty():
        root.update();time.sleep(.05)
        if time.monotonic()-t>seconds:raise TimeoutError('GUI tâche timeout')
    root.update()


def main():
    root=tk.Tk();app=SegmentationApp(root)
    root.update()
    assert app.work is not None
    gt=app.ground_truth
    y,x=np.where(gt>0)
    point=(int(np.median(x)),int(np.median(y)))
    app.run_one_click(point);wait(root,app)
    assert app.grab_mask is not None and app.refined_mask is not None
    old_rect=app.rect
    app.click_scale.set(150)
    root.update()
    assert app.grab_mask is None and app.refined_mask is None
    assert app.click_point is not None and app.rect is not None
    assert app.rect[2]*app.rect[3] > old_rect[2]*old_rect[3]
    app.retry_click();wait(root,app)
    assert app.grab_mask is not None and app.refined_mask is not None
    assert abs(app.click_zone.get()-150)<1
    out=Path(__file__).resolve().parents[1]/'results'/'smoke_export_test.png'
    old=filedialog.asksaveasfilename
    filedialog.asksaveasfilename=lambda **kwargs:str(out)
    try:app.save_result()
    finally:filedialog.asksaveasfilename=old
    with Image.open(out) as png:
        assert png.mode=='RGBA'
        assert np.array(png)[...,3].max()==255
    out.unlink(missing_ok=True)
    app.load_image(Path(__file__).resolve().parents[1]/'demo/photos/02_tasse_cafe.jpg')
    assert app.ground_truth is None
    app.run_one_click((316,210));wait(root,app)
    assert app.grab_mask is not None
    root.update()
    try:
        from PIL import ImageGrab
        ImageGrab.grab().save(Path(__file__).resolve().parents[1]/'apercu_interface_v2.png')
    except Exception as exc:print('Screenshot unavailable:',exc)
    # Réparation guidée depuis une image réelle annotée : vérifier le câblage
    # du nouveau bouton dans la GUI et la qualité sur une tige très fine.
    app.load_image(Path(__file__).resolve().parents[1]/'demo/images/produit_05.png')
    app.rect=(38,38,243,243)
    app.run_grabcut();wait(root,app)
    assert app.grab_mask is not None
    iou_avant=mask_metrics(app.grab_mask,app.ground_truth)['IoU']
    app.run_repair();wait(root,app)
    assert app.repair_used and app.refined_mask is not None
    iou_apres=mask_metrics(app.refined_mask,app.ground_truth)['IoU']
    assert iou_apres>iou_avant and iou_apres>0.96
    print(f'GUI SMOKE OK — clic photo/demo + export RGBA + réparation lampe: IoU {iou_avant:.3f} → {iou_apres:.3f}')
    root.destroy()


if __name__=='__main__':main()
