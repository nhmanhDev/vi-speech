"""Pilot v2: quét thô vi-podcast bằng AST (cửa sổ 10 s = đúng đầu vào gốc của model), lưu dần, chạy tiếp được."""
import io, json, os, sys, time, numpy as np, soundfile as sf, pyarrow.parquet as pq, torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification
sys.path.insert(0,'/home/user/vi-speech/src')
from vispeech.audio import resample
OUT='/home/user/data/tagpilot'; MAXH=float(sys.argv[1]) if len(sys.argv)>1 else 2.0
M='MIT/ast-finetuned-audioset-10-10-0.4593'
fe=ASTFeatureExtractor.from_pretrained(M); model=ASTForAudioClassification.from_pretrained(M).eval(); torch.set_num_threads(4)
lab=model.config.id2label
want={'Laughter':'cười','Giggle':'cười','Chuckle, chortle':'cười','Snicker':'cười','Belly laugh':'cười','Breathing':'thở','Gasp':'thở',
      'Sigh':'thở dài','Cough':'ho','Sniff':'sụt sịt','Throat clearing':'hắng giọng','Crying, sobbing':'khóc','Whimper':'khóc','Whispering':'thì thầm'}
ids={i:want[n] for i,n in lab.items() if n in want}
rows=[r for r in pq.read_table('/home/user/data/cand/podcast0.parquet').to_pylist() if r['tier']=='A']
prog=os.path.join(OUT,'progress.json'); st=json.load(open(prog)) if os.path.exists(prog) else {'row':0,'sec':0.0}
fd=open(os.path.join(OUT,'det2.jsonl'),'a',encoding='utf-8'); t0=time.time(); W=H=160000; HOP=80000
for ri in range(st['row'],len(rows)):
    if st['sec']/3600>=MAXH: break
    r=rows[ri]; w,sr=sf.read(io.BytesIO(r['audio']['bytes']),dtype='float32',always_2d=True); w16=resample(w.mean(1),sr,16000)
    segs=[(i,w16[i:i+W]) for i in range(0,max(1,len(w16)-W+1),HOP)] or [(0,w16)]
    for b in range(0,len(segs),8):
        x=fe([s for _,s in segs[b:b+8]],sampling_rate=16000,return_tensors='pt')
        with torch.no_grad(): p=torch.sigmoid(model(**x).logits).numpy()
        for (off,_),pp in zip(segs[b:b+8],p):
            for i,tag in ids.items():
                if pp[i]>=0.2: fd.write(json.dumps(dict(row=ri,t0=off/16000,tag=tag,cls=lab[i],p=round(float(pp[i]),3),text=r['text'][:200]),ensure_ascii=False)+'\n')
    fd.flush(); st['row']=ri+1; st['sec']+=len(w16)/16000; json.dump(st,open(prog,'w'))
    if ri%25==0: print(ri,'h=%.2f'%(st['sec']/3600),'%.0fs'%(time.time()-t0),flush=True)
print('DONE hours %.2f'%(st['sec']/3600),flush=True)
