import sys,time; sys.path.insert(0,'.')
import tools.exp21b_wordlevel as m
import pandas as pd, numpy as np
from mini_ai.transformer import BOS,EOS,SEP,PAD,init_params,loss_and_grads,clip_grads,AdamW,lr_schedule,n_params,KVModel
train_df = pd.read_parquet('data/babi/task_1-single_supporting_fact_train.parquet')
test_df = pd.read_parquet('data/babi/task_1-single_supporting_fact_test.parquet')
def pair(r): return (m.tok_words(r['passage'])+' <q> '+m.tok_words(r['question']), r['answer'].strip().lower())
train_pairs=[pair(r) for _,r in train_df.iterrows()]
test_pairs=[pair(r) for _,r in test_df.iterrows()]
all_words=set()
for q,a in train_pairs+test_pairs: all_words.update(q.split()); all_words.update(a.split())
all_words.add('<q>')
tok = m.WordTokenizer(all_words); V=len(tok)
def encode(q,a):
    qi=tok.encode(q)[:60]; ai=tok.encode(a)[:5]
    ids=[BOS]+qi+[SEP]+ai+[EOS]; mask=[0]*(2+len(qi))+[1]*(len(ai)+1)
    return ids,mask
enc=[encode(q,a) for q,a in train_pairs]
p,cfg = init_params(V, d=64, n_layer=2, n_head=4, pos='rope', seed=1)
print('params', n_params(p), flush=True)
LR=3e-3
WD=0.05   # weight decay вернули (регуляризация против переобучения на 900 примерах)
DROP=0.15 # dropout (ещё одна регуляризация, не было раньше)
opt = AdamW(p, lr=LR, wd=WD)
rng=np.random.default_rng(1)
np_rng = np.random.default_rng(42)  # для dropout
BATCH=64
steps_per_epoch = len(enc)//BATCH + 1
EPOCHS=200
total_steps = EPOCHS*steps_per_epoch
step=0; t0=time.time()

def accuracy(pairs, n=150, seed=1):
    kv=KVModel(p,cfg)
    rng_t=np.random.default_rng(seed)
    idxs=rng_t.choice(len(pairs),size=min(n,len(pairs)),replace=False)
    ok=0
    for i in idxs:
        q,gold = pairs[i]
        ids=[BOS]+tok.encode(q)[:60]+[SEP]
        out,_ = kv.generate(ids, max_new=5, temperature=0.0)
        pred=tok.decode(out).strip()
        ok += pred==gold
    return ok/len(idxs)

for ep in range(EPOCHS):
    order=rng.permutation(len(enc)); epl=0; nb=0
    for bi in range(0,len(order),BATCH):
        idxs=order[bi:bi+BATCH]; batch=[enc[i] for i in idxs]
        T=max(len(ids) for ids,_ in batch)
        X=np.full((len(batch),T),PAD,dtype=np.int64); M=np.zeros((len(batch),T))
        for i,(ids,mask) in enumerate(batch): X[i,:len(ids)]=ids; M[i,:len(mask)]=mask
        targets=np.zeros_like(X); targets[:,:-1]=X[:,1:]
        mask_t=np.zeros_like(M); mask_t[:,:-1]=M[:,1:]
        lr = lr_schedule(step, total_steps, LR, warmup=50)
        loss,grads = loss_and_grads(p,cfg,X,targets,mask_t, drop=DROP, rng=np_rng)
        clip_grads(grads,1.0); opt.step(p,grads,lr); epl+=loss; step+=1; nb+=1
    if ep%10==0 or ep==EPOCHS-1:
        train_acc = accuracy(train_pairs, n=150)
        test_acc = accuracy(test_pairs, n=150)
        print(f'ep={ep} step={step} loss={epl/nb:.3f} TRAIN_acc={train_acc:.2f} TEST_acc={test_acc:.2f} t={time.time()-t0:.0f}s', flush=True)
        if test_acc>=0.95:
            print('REACHED 95% at epoch', ep, 'time', time.time()-t0, flush=True)
            break
print('DONE total time', time.time()-t0, flush=True)
