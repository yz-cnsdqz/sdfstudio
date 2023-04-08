import os,sys, glob
import numpy as np
import subprocess



datafolder = 'outputs'
coreviews = ['CoreView_313','CoreView_315', 'CoreView_377', 'CoreView_386', 'CoreView_387', 'CoreView_390', 'CoreView_392', 'CoreView_393', 'CoreView_394']


for sub in coreviews:
    print()
    print('--------------------------------subject={}----------------------------------'.format(sub))
    datapath = os.path.join(datafolder, sub+'_*')
    frames = sorted(glob.glob(datapath))

    for idx, frame in enumerate(frames):
        pathnexts = glob.glob(os.path.join(frame, 'neus-facto/2023-*'))[0]
        configfile = os.path.join(pathnexts, 'config.yml')
        cmd = 'ns-eval --load-config={} --output-path={}.json --output-images-path={}_images'.format(configfile, 
                                                                                                     os.path.basename(frame),
                                                                                                     os.path.basename(frame))
        subprocess.call(cmd, shell=True)
