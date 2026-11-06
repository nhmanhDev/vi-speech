import json, glob, os, sys, time, random, numpy as np, soundfile as sf, torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification
sys.path.insert(0,'/home/user/vi-speech/src')
from vispeech.audio import resample
M='MIT/ast-finetuned-audioset-10-10-0.4593'; MAXH=float(sys.argv[1])
fe=ASTFeatureExtractor.from_pretrained(M); model=ASTForAudioClassification.from_pretrained(M).eval(); torch.set_num_threads(4)
lab=model.config.id2label
want={'Laughter':'cười','Giggle':'cười','Chuckle, chortle':'cười','Snicker':'cười','Belly laugh':'cười','Breathing':'thở','Gasp':'thở','Sigh':'thở dài','Cough':'ho','Sniff':'sụt sịt','Throat clearing':'hắng giọng','Crying, sobbing':'khóc','Whimper':'khóc','Whispering':'thì thầm','Music':'_nhạc'}
ids={i:want[n] for i,n in lab.items() if n in want}
fs=sorted(glob.glob('/home/user/data/lemas_probe/vi000/*.mp3')); random.Random(0).shuffle(fs)
out=open('/home/user/data/tagpilot/det4.jsonl','w'); tot=0; n=0; t0=time.time(); batch=[]; names=[]
def flush():
    global batch,names
    if not batch: return
    x=fe(batch,sampling_rate=16000,return_tensors='pt')
    with torch.no_grad(): p=torch.sigmoid(model(**x).logits).numpy()
    for f,pp in zip(names,p):
        for i,tag in ids.items():
            if pp[i]>=0.2: out.write(json.dumps(dict(file=os.path.basename(f),tag=tag,cls=lab[i],p=round(float(pp[i]),3)))+'\n')
    out.flush(); batch,names=[],[]
for f in fs:
    if tot/3600>=MAXH: break
    try: w,sr=sf.read(f,dtype='float32',always_2d=True)
    except Exception: continue
    w16=resample(w.mean(1),sr,16000)[:160000]; tot+=len(w16)/16000; n+=1; batch.append(w16); names.append(f)
    if len(batch)==16: flush()
    if n%500==0: print(n,'h=%.2f'%(tot/3600),'%.0fs'%(time.time()-t0),flush=True)
flush(); print('DONE %.2f %d'%(tot/3600,n),flush=True)
