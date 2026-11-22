"""Projet 10 — application interactive : GrabCut, clic guidé, U-Net, comparaison.

Démarrer : python app.py
Un clic lance automatiquement GrabCut puis U-Net (si modèle présent).
Les corrections au pinceau et le rectangle sont disponibles en complément.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk

from segmentation.grabcut import resize_mask, segment_grabcut, transparent_cutout
from segmentation.metrics import mask_metrics
from segmentation.one_click import propose_click_rect, segment_one_click
from segmentation.refine import load_model, refine_with_unet
from segmentation.pretrained_u2netp import PretrainedU2NetP, refine_with_pretrained
from segmentation.repair import recover_with_grabcut

ROOT = Path(__file__).resolve().parent
MAX_WORK_SIZE = 900


def checkerboard(width: int, height: int, tile: int = 16) -> Image.Image:
    bg = Image.new('RGB', (width, height), '#ffffff')
    draw = ImageDraw.Draw(bg)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if (x // tile + y // tile) % 2 == 0:
                draw.rectangle((x, y, min(width-1, x+tile-1), min(height-1,y+tile-1)), fill='#dce2e9')
    return bg


def composite_checkerboard(rgba: Image.Image) -> Image.Image:
    base = checkerboard(*rgba.size)
    base.paste(rgba, (0, 0), rgba)
    return base


def fit_image(image: Image.Image, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    size = (max(1, round(image.width*scale)), max(1,round(image.height*scale)))
    return image.resize(size,Image.Resampling.LANCZOS), scale


def to_preview(image_rgb: np.ndarray, mask: np.ndarray | None) -> Image.Image:
    if mask is None:
        return Image.fromarray(image_rgb).convert('RGB')
    return composite_checkerboard(Image.fromarray(transparent_cutout(image_rgb, mask), 'RGBA'))


def format_metric_line(grab: np.ndarray | None, refined: np.ndarray | None,
                       gt: np.ndarray | None, refined_label: str = 'GrabCut + U-Net') -> str:
    if grab is None:
        return 'Comparaison : exécutez GrabCut. IoU/Dice nécessitent un masque annoté.'
    if gt is None:
        if refined is not None:
            diff = np.count_nonzero(grab != refined) / grab.size * 100
            return f'Comparaison visuelle — {diff:.2f} % de pixels changés (ce n’est PAS un score de qualité).'
        return 'GrabCut obtenu — chargez un masque annoté pour calculer IoU et Dice.'
    a=mask_metrics(grab,gt)
    line=f"GrabCut : IoU {a['IoU']:.3f} · Dice {a['Dice']:.3f}"
    if refined is not None:
        b=mask_metrics(refined,gt)
        line+=f"    |    {refined_label} : IoU {b['IoU']:.3f} · Dice {b['Dice']:.3f}    |    ΔIoU {b['IoU']-a['IoU']:+.3f}"
    return line


class SegmentationApp:
    def __init__(self, master: tk.Tk):
        self.master=master
        self.master.title('Projet 10 | Détourage intelligent — GrabCut + U-Net')
        self.master.geometry('1430x830')
        self.master.minsize(1070,670)
        self.root_style=ttk.Style()
        try:self.root_style.theme_use('clam')
        except tk.TclError:pass
        self.root_style.configure('Title.TLabel', font=('Segoe UI',19,'bold'))
        self.root_style.configure('Subtitle.TLabel', font=('Segoe UI',10), foreground='#526377')
        self.root_style.configure('Info.TLabel', font=('Segoe UI',10,'bold'), foreground='#284A70')
        self.root_style.configure('Section.TLabelframe.Label',font=('Segoe UI',11,'bold'))
        self.mode=tk.StringVar(value='one_click')
        self.status=tk.StringVar(value='Cliquez directement sur un objet ou tracez une boîte de sélection.')
        self.metrics=tk.StringVar(value='IoU / Dice : disponibles avec des masques de référence.')
        self.demo_choice=tk.StringVar()
        self.export_choice=tk.StringVar(value='GrabCut seul')
        self.repair_used=False
        self.feather=tk.IntVar(value=0)
        self.brush_radius=tk.IntVar(value=6)
        self.click_zone=tk.DoubleVar(value=100.0)
        self.click_zone_label=tk.StringVar(value='100 %')
        self.show_masks=tk.BooleanVar(value=False)
        self.original: Image.Image | None=None
        self.work: np.ndarray | None=None
        self.path: Path | None=None
        self.ground_truth: np.ndarray | None=None
        self.rect: tuple[int,int,int,int] | None=None
        self.click_point: tuple[int,int] | None=None
        self.drag_start: tuple[int,int] | None=None
        self.drag_end: tuple[int,int] | None=None
        self.fg_points: list[tuple[int,int]]=[]
        self.bg_points: list[tuple[int,int]]=[]
        self.undo_stack: list[tuple[list[tuple[int,int]],list[tuple[int,int]]]]=[]
        self.grab_mask: np.ndarray | None=None
        self.refined_mask: np.ndarray | None=None
        self.probability_map: np.ndarray | None=None
        self.display_mapping=(0,0,1.0)
        self.photos=[]
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='segmentation')
        self.task_queue: queue.Queue=queue.Queue()
        self.busy=False
        self.model=None
        self.model_checkpoint=None
        self.pretrained_model=None
        self.refinement_choice=tk.StringVar(value="U-Net original (local)")
        self.demo_map: dict[str,Path]={}
        self._make_ui()
        self.master.after(80,self._poll_tasks)
        self.master.protocol('WM_DELETE_WINDOW',self._on_close)
        if self.demo_map:
            self.load_image(next(iter(self.demo_map.values())))

    def _make_ui(self):
        outer=ttk.Frame(self.master,padding=15)
        outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='Détourage intelligent d’objets',style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer,text='Un clic / rectangle → GrabCut → U-Net → fusion des masques → PNG transparent',
                  style='Subtitle.TLabel').pack(anchor='w',pady=(0,12))
        buttons=ttk.Frame(outer)
        buttons.pack(fill='x',pady=(0,10))
        self.action_buttons=[]
        for name,action in [('Ouvrir une image',self.open_image),('1. Lancer GrabCut',self.run_grabcut),
                            ('2. Affiner avec U-Net',self.run_unet),
                            ('3. Réparer parties manquantes (GrabCut)',self.run_repair),('Enregistrer PNG',self.save_result),
                            ('Exporter comparaison',self.save_comparison),('Réinitialiser',self.reset_annotations)]:
            btn=ttk.Button(buttons,text=name,command=action)
            btn.pack(side='left',padx=(0,7))
            self.action_buttons.append(btn)
        demos=ttk.Frame(outer)
        demos.pack(fill='x',pady=(0,10))
        ttk.Label(demos,text='Exemples :').pack(side='left',padx=(0,8))
        synthetic=sorted((ROOT/'demo/images').glob('*.png'))
        real=sorted((ROOT/'demo/photos').glob('*.jpg'))
        for i,path in enumerate(synthetic,1):self.demo_map[f'Objet synthétique {i:02d} (masque annoté)']=path
        for path in real:self.demo_map[f'Photo réelle : {path.stem[3:].replace("_"," ")}']=path
        self.demo_combo=ttk.Combobox(demos,textvariable=self.demo_choice,values=list(self.demo_map),state='readonly',width=37)
        self.demo_combo.pack(side='left',padx=(0,7))
        self.demo_combo.bind('<<ComboboxSelected>>',lambda e:self.load_demo())
        if self.demo_map:self.demo_combo.current(0)
        ttk.Button(demos,text='Charger exemple',command=self.load_demo).pack(side='left',padx=(0,16))
        ttk.Button(demos,text='Charger masque de référence',command=self.open_ground_truth).pack(side='left',padx=(0,16))
        ttk.Label(demos,text='Export :').pack(side='left',padx=(0,4))
        ttk.Combobox(demos,textvariable=self.export_choice,values=('Résultat de droite','GrabCut seul'),
                     state='readonly',width=17).pack(side='left',padx=(0,10))
        ttk.Label(demos,text='Flou alpha :').pack(side='left')
        ttk.Spinbox(demos,from_=0,to=3,textvariable=self.feather,width=3).pack(side='left',padx=4)
        model_select=ttk.Frame(outer)
        model_select.pack(fill='x',pady=(0,10))
        ttk.Label(model_select,text='Raffinement :').pack(side='left',padx=(0,8))
        self.refinement_combo=ttk.Combobox(model_select,textvariable=self.refinement_choice,values=(
            'U-Net original (local)', 'U²-Net-p pré-entraîné (fusion)',
            'U²-Net-p pré-entraîné (masque brut)'),state='readonly',width=43)
        self.refinement_combo.pack(side='left')
        self.refinement_combo.bind('<<ComboboxSelected>>',self._refinement_changed)
        ttk.Label(model_select,text='U²-Net-p : téléchargement ~5 Mo au premier usage, internet requis.',
                  style='Subtitle.TLabel').pack(side='left',padx=12)
        tools=ttk.Frame(outer)
        tools.pack(fill='x',pady=(0,10))
        ttk.Label(tools,text='Outil :',style='Info.TLabel').pack(side='left',padx=(0,6))
        for label,value in [('Un clic (GrabCut + U-Net)','one_click'),('Rectangle','rectangle'),
                            ('Garder (+)','foreground'),('Supprimer (−)','background')]:
            ttk.Radiobutton(tools,text=label,value=value,variable=self.mode).pack(side='left',padx=(2,12))
        ttk.Button(tools,text='Annuler pinceau',command=self.undo_stroke).pack(side='left',padx=4)
        ttk.Label(tools,text='Rayon :').pack(side='left',padx=(8,2))
        ttk.Spinbox(tools,from_=1,to=35,textvariable=self.brush_radius,width=3).pack(side='left')
        click_controls=ttk.Frame(outer)
        click_controls.pack(fill='x',pady=(0,10))
        ttk.Label(click_controls,text='Taille zone clic :',style='Info.TLabel').pack(side='left',padx=(0,8))
        self.click_scale=ttk.Scale(click_controls,from_=50,to=200,
                                   variable=self.click_zone,command=self._click_zone_changed,
                                   length=260)
        self.click_scale.pack(side='left',padx=(0,8))
        ttk.Label(click_controls,textvariable=self.click_zone_label,width=7).pack(side='left')
        self.reclick_button=ttk.Button(click_controls,text='Relancer ce clic',command=self.retry_click)
        self.reclick_button.pack(side='left',padx=(10,6))
        ttk.Label(click_controls,text='50 % = petite zone · 100 % = automatique · 200 % = grande zone',
                  style='Subtitle.TLabel').pack(side='left')
        ttk.Checkbutton(tools,text='Voir les contours' ,variable=self.show_masks,command=self.redraw).pack(side='right')
        columns=ttk.Frame(outer)
        columns.pack(fill='both',expand=True)
        self.canvases=[]
        headers=['1  Image originale + indications','2  GrabCut seul','3  Résultat affiné / réparation optionnelle']
        for idx,heading in enumerate(headers):
            lf=ttk.LabelFrame(columns,text=heading,padding=7,style='Section.TLabelframe')
            lf.grid(row=0,column=idx,sticky='nsew',padx=4)
            columns.columnconfigure(idx,weight=1,uniform='same')
            canvas=tk.Canvas(lf,bg='#edf0f4',highlightthickness=0)
            canvas.pack(fill='both',expand=True)
            canvas.bind('<Configure>',lambda e:self.redraw())
            self.canvases.append(canvas)
        columns.rowconfigure(0,weight=1)
        self.canvases[0].bind('<ButtonPress-1>',self.on_press)
        self.canvases[0].bind('<B1-Motion>',self.on_drag)
        self.canvases[0].bind('<ButtonRelease-1>',self.on_release)
        bottom=ttk.Frame(outer)
        bottom.pack(fill='x',pady=(12,0))
        self.progress=ttk.Progressbar(bottom,mode='indeterminate',length=80)
        self.progress.pack(side='right',padx=(10,0))
        ttk.Label(bottom,textvariable=self.metrics,style='Info.TLabel',wraplength=1200).pack(anchor='w',pady=(0,4))
        ttk.Label(bottom,textvariable=self.status,style='Subtitle.TLabel',wraplength=1200).pack(anchor='w')

    def _refinement_changed(self, event=None):
        if self.busy:return
        self.refined_mask=None
        self.probability_map=None
        self.repair_used=False
        self._update_scores()
        self.status.set("Modèle modifié : cliquez sur « 2. Affiner avec U-Net » pour recalculer le panneau droit.")
        self.redraw()

    def _click_zone_changed(self, value):
        """Met à jour la prévisualisation ; la segmentation nécessite Relancer ce clic."""
        percentage=round(float(value))
        self.click_zone_label.set(f'{percentage} %')
        if self.work is None or self.click_point is None or self.busy:
            return
        self.rect=propose_click_rect(self.work,self.click_point,zone_scale=percentage/100.0)
        self.grab_mask=None
        self.refined_mask=None
        self.probability_map=None
        self.repair_used=False
        self._update_scores()
        self.status.set('Taille du clic modifiée : le rectangle jaune indique la zone de recherche. '
                        'Cliquez « Relancer ce clic » pour recalculer GrabCut + U-Net.')
        self.redraw()

    def retry_click(self):
        if self.work is None or self.busy:return
        if self.click_point is None:
            messagebox.showinfo('Clic nécessaire', 'Choisissez « Un clic » et cliquez d’abord sur l’objet.')
            return
        self.run_one_click(self.click_point)

    def _on_close(self):
        self.executor.shutdown(wait=False,cancel_futures=True)
        self.master.destroy()

    def _set_busy(self,busy:bool,status:str=''):
        self.busy=busy
        for btn in self.action_buttons:btn.configure(state='disabled' if busy else 'normal')
        self.demo_combo.configure(state='disabled' if busy else 'readonly')
        self.click_scale.configure(state='disabled' if busy else 'normal')
        self.reclick_button.configure(state='disabled' if busy else 'normal')
        if busy:self.progress.start(12)
        else:self.progress.stop()
        if status:self.status.set(status)

    def _launch(self,description:str,job,success):
        if self.busy:return
        self._set_busy(True,description)
        fut=self.executor.submit(job)
        fut.add_done_callback(lambda done:self.task_queue.put((done,success)))

    def _poll_tasks(self):
        try:
            while True:
                future,on_success=self.task_queue.get_nowait()
                self._set_busy(False)
                try:on_success(future.result())
                except Exception as exc:
                    self.status.set(f'Échec de la segmentation : {exc}')
                    messagebox.showerror('Traitement impossible',str(exc))
                self.redraw()
        except queue.Empty:pass
        if self.master.winfo_exists():self.master.after(80,self._poll_tasks)

    def open_image(self):
        if self.busy:return
        filename=filedialog.askopenfilename(title='Choisir une photo',
                filetypes=[('Images','*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff'),('Tous les fichiers','*.*')])
        if filename:self.load_image(Path(filename))

    def load_demo(self):
        if self.busy:return
        path=self.demo_map.get(self.demo_choice.get())
        if path:self.load_image(path)

    def load_image(self,path:Path):
        if self.busy:return
        try:
            with Image.open(path) as raw:original=ImageOps.exif_transpose(raw).convert('RGB')
            if original.width<8 or original.height<8:
                raise ValueError('Image trop petite, dimensions minimales : 8×8.')
        except Exception as exc:
            messagebox.showerror('Ouverture impossible',str(exc));return
        self.path=path;self.original=original
        factor=min(1.0,MAX_WORK_SIZE/max(original.size))
        width=max(8,round(original.width*factor));height=max(8,round(original.height*factor))
        self.work=np.asarray(original.resize((width,height),Image.Resampling.LANCZOS)).copy()
        self.ground_truth=None
        ground_truth=ROOT/'demo/masks'/path.name
        if path.parent.resolve()==(ROOT/'demo/images').resolve() and ground_truth.exists():
            with Image.open(ground_truth) as m:
                self.ground_truth=(np.asarray(m.convert('L').resize((width,height),Image.Resampling.NEAREST))>127).astype(np.uint8)
        self.reset_annotations()
        source='masque annoté inclus' if self.ground_truth is not None else 'sans masque annoté'
        self.status.set(f'{path.name} — {original.width}×{original.height} — {source}. Cliquez sur l’objet ou dessinez un rectangle.')
        for label,p in self.demo_map.items():
            if p==path:self.demo_choice.set(label);break

    def open_ground_truth(self):
        if self.work is None or self.busy:return
        filename=filedialog.askopenfilename(title='Masque de référence (blanc = objet, noir = fond)',
                 filetypes=[('Masques image','*.png *.bmp *.jpg *.jpeg')])
        if not filename:return
        with Image.open(filename) as m:
            gt=np.asarray(m.convert('L').resize((self.work.shape[1],self.work.shape[0]),Image.Resampling.NEAREST))
        self.ground_truth=(gt>127).astype(np.uint8)
        self._update_scores()
        self.status.set('Masque de référence chargé. IoU/Dice recalculés (qualité de ce masque = responsabilité de l’annotateur).')

    def reset_annotations(self):
        if self.busy:return
        self.rect=None;self.click_point=None
        self.drag_start=None;self.drag_end=None
        self.fg_points=[];self.bg_points=[];self.undo_stack=[]
        self.grab_mask=None;self.refined_mask=None;self.probability_map=None
        self.repair_used=False
        self._update_scores();self.redraw()

    def undo_stroke(self):
        if self.busy:return
        if self.undo_stack:
            self.fg_points,self.bg_points=self.undo_stack.pop()
            self.refined_mask=None;self.probability_map=None
            self.status.set('Dernier coup de pinceau annulé. Relancez GrabCut.')
            self.redraw()

    def _canvas_coords(self,event):
        if self.work is None:return None
        ox,oy,scale=self.display_mapping
        if scale<=0:return None
        x=int((event.x-ox)/scale);y=int((event.y-oy)/scale)
        if 0<=x<self.work.shape[1] and 0<=y<self.work.shape[0]:return (x,y)
        return None

    def on_press(self,event):
        if self.busy:return
        point=self._canvas_coords(event)
        if point is None:return
        mode=self.mode.get()
        if mode=='rectangle':
            self.drag_start=point;self.drag_end=point
        elif mode=='one_click':
            self.click_point=point
            self.fg_points=[point];self.bg_points=[];self.undo_stack=[]
            self.rect=None;self.grab_mask=None;self.refined_mask=None
            self.run_one_click(point)
        else:
            self.undo_stack.append((self.fg_points.copy(),self.bg_points.copy()))
            self._add_point(point)

    def _add_point(self,point):
        mode=self.mode.get()
        if mode=='foreground':self.fg_points.append(point)
        elif mode=='background':self.bg_points.append(point)
        self.refined_mask=None;self.probability_map=None
        self.redraw()

    def on_drag(self,event):
        if self.busy:return
        point=self._canvas_coords(event)
        if point is None:return
        mode=self.mode.get()
        if mode=='rectangle' and self.drag_start is not None:
            self.drag_end=point;self.redraw()
        elif mode in ('foreground','background'):
            target=self.fg_points if mode=='foreground' else self.bg_points
            start=target[-1] if target else point
            count=max(1,int(np.hypot(point[0]-start[0],point[1]-start[1])/3))
            for i in range(1,count+1):
                target.append((round(start[0]+i*(point[0]-start[0])/count),round(start[1]+i*(point[1]-start[1])/count)))
            self.refined_mask=None;self.probability_map=None;self.redraw()

    def on_release(self,event):
        if self.busy:return
        if self.mode.get()=='rectangle' and self.drag_start is not None:
            p=self._canvas_coords(event) or self.drag_end
            if p is not None:
                x1,x2=sorted((self.drag_start[0],p[0]));y1,y2=sorted((self.drag_start[1],p[1]))
                if x2-x1>=8 and y2-y1>=8:
                    self.rect=(x1,y1,x2-x1,y2-y1)
                    self.click_point=None
                    self.grab_mask=None;self.refined_mask=None;self.probability_map=None
                    self.fg_points=[];self.bg_points=[];self.undo_stack=[]
                    self.status.set('Rectangle défini. Cliquez sur « 1. Lancer GrabCut ».')
            self.drag_start=None;self.drag_end=None;self.redraw()

    def _get_model(self):
        if self.model is None:
            path=ROOT/'models/unet_demo.pt'
            self.model,self.model_checkpoint=load_model(path)
        return self.model

    def _refine_selected(self, image, coarse):
        selection=self.refinement_choice.get()
        if selection.startswith('U²-Net-p'):
            if self.pretrained_model is None:
                self.pretrained_model=PretrainedU2NetP()
            fused='fusion' in selection
            return refine_with_pretrained(image,coarse,self.pretrained_model,fusion=fused)
        return refine_with_unet(image,coarse,self._get_model())

    def run_one_click(self,point):
        if self.work is None:return
        self.click_point=point
        image=self.work.copy()
        zone_scale=round(float(self.click_zone.get()))/100.0
        def job():
            coarse,rect=segment_one_click(image,point,iterations=5,zone_scale=zone_scale)
            refined=None;probs=None
            try:
                refined,probs=self._refine_selected(image,coarse)
            except FileNotFoundError:
                pass  # GrabCut fonctionne même si aucun poids U-Net n'est disponible.
            return coarse,rect,refined,probs
        def done(result):
            self.grab_mask,self.rect,self.refined_mask,self.probability_map=result
            self.repair_used=False
            self._update_scores()
            self.status.set(f'Un clic (zone {round(zone_scale*100)} %) terminé : comparer GrabCut et U-Net '
                            '(aucun gain garanti).' if self.refined_mask is not None
                            else 'Un clic : GrabCut terminé. Poids U-Net absents.')
        self._launch('Segmentation guidée par un clic…',job,done)

    def run_grabcut(self):
        if self.work is None or self.busy:return
        if self.rect is None:
            messagebox.showinfo('Sélection requise','Dessinez un rectangle, ou choisissez « Un clic » puis cliquez sur l’objet.')
            return
        image=self.work.copy();rect=self.rect
        fg=self.fg_points.copy();bg=self.bg_points.copy()
        radius=max(1,int(self.brush_radius.get()))
        def job():return segment_grabcut(image,rect,fg,bg,iterations=5,brush_radius=radius)
        def done(mask):
            self.grab_mask=mask
            self.refined_mask=None;self.probability_map=None
            self.repair_used=False
            self._update_scores()
            self.status.set('GrabCut terminé. Corrigez au pinceau si nécessaire, puis cliquez sur « 2. Affiner avec U-Net ».')
        self._launch('Segmentation GrabCut en cours…',job,done)

    def run_unet(self):
        if self.busy:return
        if self.grab_mask is None:
            messagebox.showinfo('GrabCut requis','Lancez GrabCut (rectangle ou un clic) avant U-Net.');return
        image=self.work.copy();coarse=self.grab_mask.copy()
        def job():return self._refine_selected(image,coarse)
        def done(result):
            self.refined_mask,self.probability_map=result
            self.repair_used=False
            self._update_scores()
            self.status.set(self.refinement_choice.get() + ' terminé. Comparez avec GrabCut. Export = GrabCut seul par défaut.')
        self._launch('Affinage U-Net en cours…',job,done)

    def run_repair(self):
        """Réparation optionnelle par un second GrabCut, pas par U-Net seul."""
        if self.busy: return
        if self.grab_mask is None:
            messagebox.showinfo('GrabCut requis', 'Lancez une première segmentation avant la réparation.')
            return
        image = self.work.copy()
        coarse = self.grab_mask.copy()
        def job():
            repaired, changed = recover_with_grabcut(image, coarse)
            if not changed:
                return coarse.copy(), False, None
            # Réparation indépendante de U-Net ; aucun modèle requis.
            return repaired, True, None
        def done(result):
            repaired, changed, probability = result
            if changed:
                self.refined_mask = repaired
                self.repair_used = True
                self.probability_map = probability
                self.export_choice.set('Résultat de droite')
            # Si l'option refuse la réparation, ne pas écraser un résultat
            # précédent de U-Net qui serait éventuellement utile.
            self._update_scores()
            self.status.set('Réparation guidée par GrabCut effectuée. Les pixels retrouvés NE sont pas attribués à U-Net.'
                            if changed else
                            'Réparation non appliquée (séparabilité couleur insuffisante ou expansion risquée).')
        self._launch('Seconde segmentation GrabCut guidée en cours…', job, done)

    def _update_scores(self):
        self.metrics.set(format_metric_line(self.grab_mask,self.refined_mask,self.ground_truth,
                         'GrabCut guidé (option)' if self.repair_used else self.refinement_choice.get()) +
                         (' | Colonne droite : réparation GrabCut guidée' if self.repair_used else ''))

    def _selected_mask(self):
        if self.export_choice.get()=='GrabCut seul':return self.grab_mask
        return self.refined_mask if self.refined_mask is not None else self.grab_mask

    def save_result(self):
        if self.original is None or self.busy:return
        mask=self._selected_mask()
        if mask is None:
            messagebox.showinfo('Aucun résultat','Effectuez d’abord une segmentation GrabCut.');return
        output=filedialog.asksaveasfilename(title='Enregistrer PNG transparent',defaultextension='.png',
                 initialfile=(self.path.stem if self.path else 'objet')+'_detoure.png',
                 filetypes=[('PNG transparent','*.png')])
        if not output:return
        binary=resize_mask(mask,self.original.height,self.original.width)
        rgba=transparent_cutout(np.asarray(self.original),binary)
        radius=min(3,max(0,int(self.feather.get())))
        if radius:
            rgba[:,:,3]=cv2.GaussianBlur(rgba[:,:,3],(0,0),float(radius))
        Image.fromarray(rgba,'RGBA').save(output)
        self.status.set(f'PNG enregistré : {output}. Les couleurs RGB originales sont conservées.')

    def save_comparison(self):
        if self.work is None or self.grab_mask is None or self.busy:
            messagebox.showinfo('Aucun résultat','Lancez GrabCut pour créer une comparaison.');return
        output=filedialog.asksaveasfilename(title='Exporter la comparaison',defaultextension='.png',
                 initialfile=(self.path.stem if self.path else 'objet')+'_comparaison.png',
                 filetypes=[('PNG','*.png')])
        if not output:return
        h,w=self.work.shape[:2]
        previews=[Image.fromarray(self.work),to_preview(self.work,self.grab_mask),to_preview(self.work,self.refined_mask)]
        titles=['Image originale','GrabCut seul',
                ('Réparation GrabCut guidée' if self.repair_used else self.refinement_choice.get())]
        strip=Image.new('RGB',(w*3,h+72),'#f5f7fa')
        for i,img in enumerate(previews):
            strip.paste(img,(i*w,54))
            ImageDraw.Draw(strip).text((i*w+14,18),titles[i],fill='#152941')
        strip.save(output)
        self.status.set(f'Comparaison enregistrée : {output}')

    def _preview(self,image:Image.Image, canvas:tk.Canvas, index:int):
        width=max(80,canvas.winfo_width());height=max(80,canvas.winfo_height())
        img,scale=fit_image(image,max(10,width-24),max(10,height-24))
        if img.mode=='RGBA':img=composite_checkerboard(img)
        ox=(width-img.width)//2;oy=(height-img.height)//2
        photo=ImageTk.PhotoImage(img)
        canvas.delete('all')
        canvas.create_image(ox,oy,image=photo,anchor='nw')
        self.photos.append(photo)
        if index==0:self.display_mapping=(ox,oy,scale)

    def redraw(self):
        if self.work is None or not hasattr(self,'canvases'):return
        self.photos=[]
        source=Image.fromarray(self.work.copy())
        if self.show_masks.get() and self.grab_mask is not None:
            cnt,_=cv2.findContours(self.grab_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            overlay=np.array(source)
            cv2.drawContours(overlay,cnt,-1,(40,190,90),2)
            source=Image.fromarray(overlay)
        draw=ImageDraw.Draw(source)
        r=self.rect
        if self.drag_start is not None and self.drag_end is not None:
            x1,y1=self.drag_start;x2,y2=self.drag_end
            r=(min(x1,x2),min(y1,y2),abs(x2-x1),abs(y2-y1))
        if r:
            x,y,w,h=r
            draw.rectangle((x,y,x+w,y+h),outline='#f5d130',width=3)
        radius=max(1,int(self.brush_radius.get()))
        for x,y in self.fg_points:
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill='#4bea85',outline='#185a38',width=1)
        for x,y in self.bg_points:
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill='#fc6678',outline='#8d213a',width=1)
        self._preview(source,self.canvases[0],0)
        for idx,mask in ((1,self.grab_mask),(2,self.refined_mask)):
            if mask is None:img=checkerboard(self.work.shape[1],self.work.shape[0])
            else:img=to_preview(self.work,mask)
            self._preview(img,self.canvases[idx],idx)


def main():
    root=tk.Tk()
    SegmentationApp(root)
    root.mainloop()


if __name__=='__main__':main()
