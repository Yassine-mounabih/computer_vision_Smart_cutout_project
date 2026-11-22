"""Comparaison visuelle sur photographies sans vérité terrain (PAS de scores IoU/Dice)."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
import numpy as np
from app import to_preview
from segmentation.one_click import segment_one_click
from segmentation.refine import load_model,refine_with_unet
from segmentation.repair import recover_with_grabcut

ROOT=Path(__file__).resolve().parents[1]
POINTS={
    '01_chat_Chelsea.jpg':(232,160),
    '02_tasse_cafe.jpg':(316,210),
    '03_astronaute.jpg':(190,292),
    '04_fusee.jpg':(321,210),
}


def main():
    model,info=load_model(ROOT/'models/unet_demo.pt')
    dest=ROOT/'results/photos_qualitatives'
    dest.mkdir(parents=True,exist_ok=True)
    lines=['Comparaison QUALITATIVE, sans masque de référence.','Aucun IoU/Dice calculé et aucune promesse de perfection.','']
    for path in sorted((ROOT/'demo/photos').glob('*.jpg')):
        point=POINTS.get(path.name)
        if not point:continue
        image=np.array(Image.open(path).convert('RGB'))
        try:
            coarse,rect=segment_one_click(image,point)
            refined,_=refine_with_unet(image,coarse,model)
            guided, applied = recover_with_grabcut(image,coarse)
            pics=[Image.fromarray(image),to_preview(image,coarse),to_preview(image,refined),to_preview(image,guided)]
            width,height=image.shape[1],image.shape[0]
            comparison=Image.new('RGB',(width*4,height+40),'#f5f7fa')
            d=ImageDraw.Draw(comparison)
            for i,(pic,name) in enumerate(zip(pics,['Original','GrabCut seul','U-Net + fusion','GrabCut guide (option)'])):
                comparison.paste(pic,(i*width,40))
                d.text((i*width+10,13),name,fill='#1b2b3b')
            comparison.save(dest/(path.stem+'_comparaison.png'))
            changes=int(np.count_nonzero(coarse!=refined))
            lines.append(f'{path.name}: clic {point}, boîte {rect}, U-Net pixels modifiés={changes}; GrabCut guide appliqué={applied}; jugement visuel nécessaire.')
            print(f'{path.name}: OK, U-Net pixels modifiés={changes}, GrabCut guidé={applied}')
        except Exception as exc:
            lines.append(f'{path.name}: ÉCHEC {exc}')
            print(f'{path.name}: ÉCHEC {exc}')
    (dest/'LIRE_MOI.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
