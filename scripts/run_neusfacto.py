import os,sys, glob
import numpy as np
import subprocess



datafolder = '/mnt/hdd/datasets/ZJUMocap_SDFStudio'
coreviews = ['CoreView_313','CoreView_315', 'CoreView_377', 'CoreView_386', 'CoreView_387', 'CoreView_390', 'CoreView_392', 'CoreView_393', 'CoreView_394']


for sub in coreviews:
    print()
    print('--------------------------------subject={}----------------------------------'.format(sub))
    datapath = os.path.join(datafolder, sub, 'train')
    frames = glob.glob(datapath+'/*')
    
    for frame in frames:
        tidx = os.path.basename(frame)
        expname = sub+'_'+tidx
        cmd = 'ns-train neus-facto --pipeline.model.sdf-field.inside-outside=False --pipeline.datamanager.train-num-rays-per-batch=1024 --trainer.max-num-iterations=20000 --trainer.save-only-latest-checkpoint=True --experiment-name {} --vis tensorboard sdfstudio-data --data {} --include_foreground_mask True --skip_every_for_val_split 2 --train_val_no_overlap True'.format(expname, frame)
        subprocess.call(cmd, shell=True)