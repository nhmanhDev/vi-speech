import json, glob, os, sys, time, numpy as np, soundfile as sf, torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification
sys.path.insert(0,'/home/user/vi-speech/src')
from vispeech.audio import resample
M='MIT/ast-finetuned-audioset-10-10-0.4593'; MAXH=float(sys.argv[1])
fe=ASTFeatureExtractor.from_pretrained(M); model=ASTForAudioClassification.from_pretrained(M).eval(); torch.set_num_threads(4)
lab=model.config.id2label
want={'Laughter':'cười','Giggle':'cười','Chuckle, chortle':'cười','Snicker':'cười','Belly laugh':'cười','Breathing':'thở','Gasp':'thở','Sigh':'thở dài','Cough':'ho','Sniff':'sụt sịt','Throat clearing':'hắng giọng','Crying, sobbing':'khóc','Whimper':'khóc','Whispering':'thì thầm','Music':'_nhạc','Applause':'_vỗ tay'}
ids={i:want[n] for i,n in lab.items() if n in want}
D='/home/user/data/yodas_probe'; out=open('/home/user/data/tagpilot/det3.jsonl','w'); tot=0; t0=time.time(); W=160000; HOP=80000
for f in sorted(glob.glob(D+'/*.flac'), key=lambda x:int(os.path.basename(x).split('.')[0])):
    if tot/3600>=MAXH: break
    w,sr=sf.read(f,dtype='float32',always_2d=True); w16=resample(w.mean(1),sr,16000); tot+=len(w16)/16000
    for b in range(0,len(w16)-W+1,HOP*8):
        segs=[w16[i:i+W] for i in range(b,min(b+HOP*8,len(w16)-W+1),HOP)]
        x=fe(segs,sampling_rate=16000,return_tensors='pt')
        with torch.no_grad(): p=torch.sigmoid(model(**x).logits).numpy()
        for j,pp in enumerate(p):
            for i,tag in ids.items():
                if pp[i]>=0.2: out.write(json.dumps(dict(file=os.path.basename(f),t0=(b+j*HOP)/16000,tag=tag,cls=lab[i],p=round(float(pp[i]),3)))+'\n')
    out.flush(); print(os.path.basename(f),'h=%.2f'%(tot/3600),'%.0fs'%(time.time()-t0),flush=True)
print('DONE %.2f'%(tot/3600),flush=True)
