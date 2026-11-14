# Rerun for supervisor comment: "Confirm whether severity words were removed before generating all three
# feature groups (TF-IDF, emotion probabilities, keyword counts)."
#
# Finding from the code check: in ablation_clean.py, repeated_splits_wilcoxon_chi2.py,
# bugzilla_masking_sensitivity.py and final_metrics_figures.py the Bugzilla text was masked for TF-IDF and keyword
# counts, but the emotion probabilities were loaded from emo_probs_bz_*.npy, which had been computed on the
# ORIGINAL (unmasked) text. This script recomputes the emotion probabilities on masked text and reruns every
# Bugzilla comparison that uses distress features, with ONE seed set (42-46) everywhere.
#
# GitHub is not affected (no masking is applied there), so its results stay as they are.
#
# Run in Colab with a GPU (T4/L4). About 25-40 min. Paste the whole printed output back.

from google.colab import drive
drive.mount('/content/drive')
import re, json, hashlib, numpy as np, pandas as pd, scipy.sparse as sp, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import chi2, kruskal, wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import torch
from transformers import pipeline

P = '/content/drive/MyDrive/Colab Notebooks/'
SEEDS = [42, 43, 44, 45, 46]
CLS = ['Critical', 'High', 'Low']; CID = {k: i for i, k in enumerate(CLS)}
SEV = r'\b(blockers?|critical|criticals|major|majors|minor|minors|trivial|normal|severity|sev|enhancement|regression)\b'
mask = lambda s: re.sub(r'\s+', ' ', re.sub(SEV, ' ', s)).strip()
LEX = {'frustration': ['frustrating', 'annoying', 'broken', 'failed', 'crash', 'error'],
       'confusion': ['confused', 'unclear', 'dont understand', 'unknown', 'complicated'],
       'anxiety': ['worried', 'anxious', 'fear', 'unsafe', 'security'],
       'urgency': ['urgent', 'immediately', 'asap', 'critical', 'important'],
       'helplessness': ['stuck', 'unable', 'cannot', 'helpless', 'nothing works'],
       'anger': ['hate', 'terrible', 'worst', 'angry', 'useless']}
EMO = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
dds = lambda texts: np.array([[sum(t.lower().count(w) for w in ws) for ws in LEX.values()] for t in texts])
xgb = lambda s: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                              tree_method='hist', device='cuda', random_state=s)
md5 = lambda f: hashlib.md5(open(f, 'rb').read()).hexdigest()[:10]

# ---------- 0. data + dataset version ----------
d = {n: pd.read_csv(P + f'clean_bugzilla_{n}.csv') for n in ('train', 'val', 'test')}
for n in d: d[n]['text'] = d[n]['text'].fillna('').astype(str); d[n]['mtext'] = d[n]['text'].map(mask)
y = {n: d[n]['y'].map(CID).values for n in d}
VERSION = {n: md5(P + f'clean_bugzilla_{n}.csv') for n in d}
print('DATASET VERSION (md5 of clean_bugzilla_*.csv):', VERSION)
print('Bugzilla class counts [Critical, High, Low]:', {n: d[n]['y'].value_counts().reindex(CLS).tolist() for n in d})
g = {n: pd.read_csv(P + f'clean_github_{n}.csv') for n in ('train', 'val', 'test')}
print('GitHub class counts [Fast, Medium, Slow]:', {n: g[n]['y'].value_counts().reindex(['Fast', 'Medium', 'Slow']).tolist() for n in g})
print('GitHub version (md5):', {n: md5(P + f'clean_github_{n}.csv') for n in g})
print('GitHub columns:', list(g['train'].columns))
raw_bz = pd.read_csv(P + 'Bugzilla_sev.csv'); print('Bugzilla raw columns:', list(raw_bz.columns))
print('Bugzilla raw severity counts:', raw_bz['Severity'].value_counts().to_dict())
for c in raw_bz.columns:
    if any(k in c.lower() for k in ('product', 'component', 'date', 'creat', 'open', 'time')):
        s = raw_bz[c]; print(f'  {c}: e.g. {s.dropna().astype(str).head(3).tolist()}  n_unique={s.nunique()}')
raw_gh = pd.read_csv(P + 'GitHub_bugs.csv'); print('GitHub raw columns:', list(raw_gh.columns))
_t = pd.to_datetime(raw_gh['Created'], errors='coerce', utc=True); print('GitHub created range:', _t.min(), '->', _t.max())
for c in raw_gh.columns:
    if any(k in c.lower() for k in ('repo', 'project', 'url', 'owner')):
        print(f'  {c}: n_unique={raw_gh[c].nunique()}  top={raw_gh[c].value_counts().head(5).to_dict()}')

# ---------- 1. emotion probabilities on MASKED text ----------
clf = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None,
               device=0 if torch.cuda.is_available() else -1, truncation=True, max_length=256)
def emo(texts, bs=64):
    out = clf(list(texts), batch_size=bs)
    return np.array([[{r['label']: r['score'] for r in o}[e] for e in EMO] for o in out])
_chk = emo(d['train']['text'].iloc[:200]); _old = np.load(P + 'emo_probs_bz_train.npy')[:200]
print('column-order check vs stored original probs (should be ~0): %.4f' % np.abs(_chk - _old).mean())
Eo, Em = {}, {}
for n in d:
    Eo[n] = np.load(P + f'emo_probs_bz_{n}.npy')
    f = P + f'emo_probs_bzmask_{n}.npy'
    try: Em[n] = np.load(f); print('loaded', f)
    except FileNotFoundError: Em[n] = emo(d[n]['mtext']); np.save(f, Em[n]); print('saved', f)
ch = np.abs(Em['train'] - Eo['train']).sum(1)
print('emotion vectors changed by masking (train): %.1f%% of texts; mean L1 change %.4f; top emotion changed %.1f%%'
      % (100 * (ch > 1e-6).mean(), ch.mean(), 100 * (Em['train'].argmax(1) != Eo['train'].argmax(1)).mean()))

def feats(text_col, E):
    K = {n: dds(d[n][text_col]) for n in d}
    return {'emo': E, 'kw': {n: np.c_[K[n], K[n].sum(1)] for n in d}}

def build(text_col, blocks, keys, use_text=True):
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(d['train'][text_col])
    dense = lambda n: np.hstack([blocks[k][n] for k in keys]) if keys else None
    sc = StandardScaler().fit(dense('train')) if keys else None
    X = {}
    for n in d:
        parts = ([tf.transform(d[n][text_col])] if use_text else []) + ([sp.csr_matrix(sc.transform(dense(n)))] if keys else [])
        X[n] = sp.hstack(parts).tocsr()
    return X

def run(X):
    sw = compute_sample_weight('balanced', y['train']); P_, F, A, AUC = {}, [], [], []
    for s in SEEDS:
        m = xgb(s).fit(X['train'], y['train'], sample_weight=sw); pr = m.predict_proba(X['test']); p = pr.argmax(1)
        P_[s] = (p, pr); F.append(f1_score(y['test'], p, average='macro')); A.append(accuracy_score(y['test'], p))
        AUC.append(roc_auc_score(label_binarize(y['test'], classes=[0, 1, 2]), pr, average='macro'))
    return dict(preds=P_, f1=np.array(F), acc=np.array(A), auc=np.array(AUC))

rng = np.random.default_rng(0); IDX = [rng.integers(0, len(y['test']), len(y['test'])) for _ in range(1000)]
def boot(ra, rb):  # matched-seed bootstrap: mean over seeds of (b - a) in each resample
    dd = np.array([np.mean([f1_score(y['test'][i], rb['preds'][s][0][i], average='macro') -
                            f1_score(y['test'][i], ra['preds'][s][0][i], average='macro') for s in SEEDS]) for i in IDX])
    return dd.mean(), np.percentile(dd, 2.5), np.percentile(dd, 97.5)

RECORD = []
def rec(item, text, feats_, seeds, note=''):
    RECORD.append(dict(item=item, dataset='bugzilla ' + json.dumps(VERSION), split='stratified 70/15/15, seed 42',
                       model_seeds=seeds, masking=text, features=feats_, note=note))

# ---------- 2. masking sensitivity (original vs masked, all features consistent) ----------
print('\n=== MASKING SENSITIVITY (XGBoost, 5 seeds) ===')
R = {}
for tag, col, E in (('original', 'text', Eo), ('masked', 'mtext', Em)):
    B = feats(col, E)
    R[tag, 'base'] = run(build(col, B, []))
    R[tag, 'dist'] = run(build(col, B, ['emo', 'kw']))
    dlt = boot(R[tag, 'base'], R[tag, 'dist'])
    print(f"{tag:9s} TF-IDF {R[tag,'base']['f1'].mean():.4f}±{R[tag,'base']['f1'].std():.4f}  "
          f"+distress {R[tag,'dist']['f1'].mean():.4f}±{R[tag,'dist']['f1'].std():.4f}  "
          f"diff {dlt[0]:+.4f} [{dlt[1]:+.4f}, {dlt[2]:+.4f}]")
    rec(f'Masking paragraph ({tag})', tag, 'TF-IDF; TF-IDF+emotion+keywords', '42-46')

# ---------- 3. ablation A1-A5, masked text, masked emotion ----------
print('\n=== ABLATION (masked text, masked emotion, 5 seeds) ===')
B = feats('mtext', Em); ABL = {}
cfg = {'A1 TF-IDF': (True, []), 'A2 +emotion': (True, ['emo']), 'A3 +keywords': (True, ['kw']),
       'A4 +both': (True, ['emo', 'kw']), 'A5 distress only': (False, ['emo', 'kw'])}
for c, (ut, k) in cfg.items():
    ABL[c] = run(build('mtext', B, k, ut))
for c in cfg:
    r = ABL[c]; line = f"{c:17s} acc {r['acc'].mean():.4f}  F1 {r['f1'].mean():.4f}±{r['f1'].std():.4f}"
    if c != 'A1 TF-IDF':
        dlt = boot(ABL['A1 TF-IDF'], r); line += f"  diff {dlt[0]:+.4f} [{dlt[1]:+.4f}, {dlt[2]:+.4f}]"
    print(line)
rec('Table XI (ablation, Bugzilla)', 'masked (TF-IDF, emotion, keywords)', 'A1-A5', '42-46')

# ---------- 4. ROC-AUC, per-class, confusion, McNemar (masked) ----------
print('\n=== ROC-AUC / PER-CLASS / CONFUSION / McNEMAR (masked) ===')
base, dist = ABL['A1 TF-IDF'], ABL['A4 +both']
print('ROC-AUC XGB %.4f±%.4f   XGB+distress %.4f±%.4f' % (base['auc'].mean(), base['auc'].std(), dist['auc'].mean(), dist['auc'].std()))
p0, p1 = base['preds'][42][0], dist['preds'][42][0]
print(classification_report(y['test'], p1, target_names=CLS, digits=3))
cm = confusion_matrix(y['test'], p1); print('confusion (counts, rows=true):\n', cm)
ca, cb = p0 == y['test'], p1 == y['test']; n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
st = (abs(n01 - n10) - 1) ** 2 / (n01 + n10); print('McNemar +distress right/base wrong %d vs %d, p=%.4f' % (n01, n10, 1 - chi2.cdf(st, 1)))
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for a, (nm, p) in zip(ax, (('XGBoost, TF-IDF', p0), ('XGBoost, TF-IDF + distress', p1))):
    c = confusion_matrix(y['test'], p); cn = c / c.sum(1, keepdims=True)
    a.imshow(cn, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3): a.text(j, i, f'{cn[i,j]:.2f}\n({c[i,j]})', ha='center', va='center', color='white' if cn[i, j] > .5 else 'black', fontsize=9)
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(CLS); a.set_yticklabels(CLS)
    a.set_xlabel('Predicted'); a.set_ylabel('True'); a.set_title(nm, fontsize=10)
plt.tight_layout(); plt.savefig(P + 'fig_confusion_bugzilla_masked.png', dpi=300); print('saved fig_confusion_bugzilla_masked.png')
rec('Table IX, per-class table, Fig. 7 (Bugzilla), McNemar', 'masked (all features)', 'XGB vs XGB+distress', '42-46 (AUC); 42 (per-class, confusion, McNemar)')

# ---------- 5. emotion-severity association on masked emotion ----------
print('\n=== ASSOCIATION (Kruskal-Wallis, all Bugzilla reports) ===')
Eall_o = np.vstack([Eo[n] for n in d]); Eall_m = np.vstack([Em[n] for n in d]); yall = np.concatenate([y[n] for n in d])
for nm, E in (('original', Eall_o), ('masked', Eall_m)):
    out = []
    for j, e in enumerate(EMO):
        H, p = kruskal(*[E[yall == k, j] for k in range(3)]); out.append(f'{e} eps2={H/(len(yall)-1):.3f}')
    print(nm, '; '.join(out))
rec('Table X / Fig. 4 (association, Bugzilla)', 'original and masked', 'emotion probabilities', 'n/a')

# ---------- 6. ten repeated splits, masked emotion ----------
print('\n=== REPEATED SPLITS (10 stratified 70/15/15 splits of pooled masked Bugzilla) ===')
df = pd.concat([d[n] for n in d], ignore_index=True); Ea = Eall_m; K = dds(df.mtext); Da = np.c_[Ea, K, K.sum(1)]; ya = yall
rows = []
for rep in range(10):
    tr, tmp = train_test_split(np.arange(len(df)), test_size=.30, stratify=ya, random_state=rep)
    _, te = train_test_split(tmp, test_size=.50, stratify=ya[tmp], random_state=rep)
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(df.mtext.iloc[tr])
    Ttr, Tte = tf.transform(df.mtext.iloc[tr]), tf.transform(df.mtext.iloc[te])
    sc = StandardScaler().fit(Da[tr]); Dtr, Dte = sc.transform(Da[tr]), sc.transform(Da[te]); sw = compute_sample_weight('balanced', ya[tr])
    r = dict(rep=rep)
    r['LogReg'] = f1_score(ya[te], LogisticRegression(max_iter=1000, class_weight='balanced').fit(Ttr, ya[tr]).predict(Tte), average='macro')
    r['XGB'] = f1_score(ya[te], xgb(rep).fit(Ttr, ya[tr], sample_weight=sw).predict(Tte), average='macro')
    r['XGB+distress'] = f1_score(ya[te], xgb(rep).fit(sp.hstack([Ttr, sp.csr_matrix(Dtr)]).tocsr(), ya[tr], sample_weight=sw)
                                 .predict(sp.hstack([Tte, sp.csr_matrix(Dte)]).tocsr()), average='macro')
    rows.append(r); print(rep, {k: round(v, 4) for k, v in r.items() if k != 'rep'})
rs = pd.DataFrame(rows)
for k in ('LogReg', 'XGB', 'XGB+distress'): print(f'{k}: {rs[k].mean():.4f}±{rs[k].std():.4f}')
for a, b in (('XGB', 'XGB+distress'), ('LogReg', 'XGB')):
    dd = rs[b] - rs[a]; print(f'{b} - {a}: mean {dd.mean():+.4f}, Wilcoxon p={wilcoxon(rs[b], rs[a]).pvalue:.4f}, wins {(dd>0).sum()}/10')
rec('Table XII / Fig. 5 (repeated splits, Bugzilla)', 'masked (all features)', 'LogReg, XGB, XGB+distress', 'seed = split index 0-9',
    'splits of one corpus, not independent datasets')

pd.DataFrame(RECORD).to_csv(P + 'run_record_bugzilla_masked.csv', index=False)
print('\nsaved run_record_bugzilla_masked.csv')
