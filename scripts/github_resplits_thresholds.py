# Supervisor round 3:
#  (A) audit records for the conference pipeline (balanced files, 80:20 split overlap)
#  (B) GitHub provenance check
#  (C) GitHub ten resplits with tertile thresholds recomputed inside EACH training partition
#  (D) per-class precision/recall/F1 on the main split (Bugzilla masked, GitHub) + saved predictions
# Everything is written to  MyDrive/Colab Notebooks/repro/  for the review repository.
# Colab, GPU if available. Paste the whole printed output back.

from google.colab import drive
drive.mount('/content/drive')
import os, re, json, numpy as np, pandas as pd, scipy.sparse as sp, torch
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier

P = '/content/drive/MyDrive/Colab Notebooks/'
R = P + 'repro/'; os.makedirs(R, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LEX = {'frustration': ['frustrating', 'annoying', 'broken', 'failed', 'crash', 'error'],
       'confusion': ['confused', 'unclear', 'dont understand', 'unknown', 'complicated'],
       'anxiety': ['worried', 'anxious', 'fear', 'unsafe', 'security'],
       'urgency': ['urgent', 'immediately', 'asap', 'critical', 'important'],
       'helplessness': ['stuck', 'unable', 'cannot', 'helpless', 'nothing works'],
       'anger': ['hate', 'terrible', 'worst', 'angry', 'useless']}
dds = lambda texts: np.array([[sum(t.lower().count(w) for w in ws) for ws in LEX.values()] for t in texts])
xgb = lambda s: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                              tree_method='hist', device=DEV, random_state=s)
tfidf = lambda: TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)

# ---------------- (A) audit of the conference-era files ----------------
print('=== (A) AUDIT OF CONFERENCE-ERA BALANCED FILES ===')
for f in ('so_severity_balanced.csv', 'github_severity_balanced.csv', 'bugzilla_severity_balanced.csv'):
    try: b = pd.read_csv(P + f)
    except FileNotFoundError: print(f, 'not found'); continue
    tc = next(c for c in ('text', 'clean_text', 'Text', 'Body', 'Description') if c in b.columns)
    lc = next((c for c in ('severity', 'Severity', 'label', 'severity_label') if c in b.columns), None)
    t = b[tc].fillna('').astype(str)
    line = f'{f}: rows={len(b)} unique_texts={t.nunique()} repeated_rows={len(b)-t.nunique()}'
    if lc:
        line += f' class_counts={b[lc].value_counts().to_dict()}'
        tr, te = train_test_split(b, test_size=0.2, random_state=42, stratify=b[lc])   # the conference 80:20 split
        line += f' | 80:20 seed-42 split: train={len(tr)} test={len(te)} identical_texts_in_both={len(set(tr[tc].astype(str)) & set(te[tc].astype(str)))}'
    print(line)

# ---------------- (B) GitHub provenance check ----------------
print('\n=== (B) GITHUB PROVENANCE CHECK ===')
gr = pd.read_csv(P + 'GitHub_bugs.csv')
print('Issue id range:', gr['Issue id'].min(), '->', gr['Issue id'].max(), ' n =', len(gr), ' unique ids =', gr['Issue id'].nunique())
txt = (gr['Summary'].fillna('') + ' ' + gr['Description'].fillna('')).str.lower()
for k in ('vscode', 'vs code', 'visual studio code', 'copilot', 'extension host', 'workbench', 'monaco'):
    print(f'  texts mentioning "{k}": {txt.str.contains(k, regex=False).mean()*100:.1f}%')
print('  sample titles:', gr['Summary'].dropna().head(8).tolist())

# ---------------- (C) GitHub resplits, thresholds per training partition ----------------
print('\n=== (C) GITHUB TEN RESPLITS, TERTILE THRESHOLDS FROM EACH TRAINING PARTITION ===')
parts = [pd.read_csv(P + f'clean_github_{n}.csv') for n in ('train', 'val', 'test')]
df = pd.concat(parts, ignore_index=True); df['text'] = df['text'].fillna('').astype(str)
E = np.vstack([np.load(P + f'emo_probs_gh_{n}.npy') for n in ('train', 'val', 'test')])
assert len(E) == len(df) and 'days' in df.columns
K = dds(df.text); D = np.c_[E, K, K.sum(1)]; days = df['days'].values
rows, split_rec = [], []
for rep in range(10):
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=rep)          # split first (labels do not exist yet)
    va, te = train_test_split(tmp, test_size=0.50, random_state=rep)
    q1, q2 = np.quantile(days[tr], [1/3, 2/3])                                  # thresholds from THIS training partition only
    y = np.where(days <= q1, 0, np.where(days <= q2, 1, 2))                      # applied unchanged to val and test
    tf = tfidf().fit(df.text.iloc[tr]); Ttr, Tte = tf.transform(df.text.iloc[tr]), tf.transform(df.text.iloc[te])
    sc = StandardScaler().fit(D[tr]); Dtr, Dte = sc.transform(D[tr]), sc.transform(D[te])
    sw = compute_sample_weight('balanced', y[tr])
    pl = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Ttr, y[tr]).predict(Tte)
    px = xgb(rep).fit(Ttr, y[tr], sample_weight=sw).predict(Tte)
    pd_ = xgb(rep).fit(sp.hstack([Ttr, sp.csr_matrix(Dtr)]).tocsr(), y[tr], sample_weight=sw).predict(sp.hstack([Tte, sp.csr_matrix(Dte)]).tocsr())
    r = dict(rep=rep, q1_days=round(q1, 3), q2_days=round(q2, 3),
             test_counts=np.bincount(y[te], minlength=3).tolist(),
             LogReg=f1_score(y[te], pl, average='macro'), XGB=f1_score(y[te], px, average='macro'), **{'XGB+distress': f1_score(y[te], pd_, average='macro')})
    rows.append(r); print(rep, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != 'rep'})
    pd.DataFrame({'row': te, 'y_true': y[te], 'LogReg': pl, 'XGB': px, 'XGB+distress': pd_}).to_csv(R + f'gh_resplit{rep}_test_predictions.csv', index=False)
    split_rec.append(dict(rep=rep, train=tr.tolist(), val=va.tolist(), test=te.tolist(), q1=q1, q2=q2))
json.dump(split_rec, open(R + 'gh_resplits_indices_thresholds.json', 'w'))
rs = pd.DataFrame(rows); rs.to_csv(R + 'gh_resplits_results.csv', index=False)
for k in ('LogReg', 'XGB', 'XGB+distress'): print(f'{k}: {rs[k].mean():.4f}±{rs[k].std():.4f}')
for a, b in (('XGB', 'XGB+distress'), ('LogReg', 'XGB')):
    dd = rs[b] - rs[a]; print(f'{b} - {a}: mean {dd.mean():+.4f}, Wilcoxon p={wilcoxon(rs[b], rs[a]).pvalue:.4f}, wins {(dd>0).sum()}/10')
print('threshold range over resplits: q1 %.2f-%.2f days, q2 %.2f-%.2f days' % (rs.q1_days.min(), rs.q1_days.max(), rs.q2_days.min(), rs.q2_days.max()))

# ---------------- (D) per-class P/R/F1 on the main split + saved predictions ----------------
print('\n=== (D) PER-CLASS PRECISION / RECALL / F1, MAIN SPLIT (seed 42) ===')
SEV = r'\b(blockers?|critical|criticals|major|majors|minor|minors|trivial|normal|severity|sev|enhancement|regression)\b'
def main_split(pre, classes, emo_tag, mask):
    d = {n: pd.read_csv(P + f'clean_{pre}_{n}.csv') for n in ('train', 'test')}
    for n in d:
        d[n]['text'] = d[n]['text'].fillna('').astype(str)
        if mask: d[n]['text'] = d[n]['text'].map(lambda s: re.sub(r'\s+', ' ', re.sub(SEV, ' ', s)).strip())
    y = {n: d[n]['y'].map({c: i for i, c in enumerate(classes)}).values for n in d}
    Em = {n: np.load(P + f'emo_probs_{emo_tag}_{n}.npy') for n in d}
    Kd = {n: dds(d[n].text) for n in d}; Dd = {n: np.c_[Em[n], Kd[n], Kd[n].sum(1)] for n in d}
    sc = StandardScaler().fit(Dd['train']); tf = tfidf().fit(d['train'].text)
    T = {n: tf.transform(d[n].text) for n in d}; sw = compute_sample_weight('balanced', y['train'])
    out = {'y_true': y['test']}
    out['XGB'] = xgb(42).fit(T['train'], y['train'], sample_weight=sw).predict(T['test'])
    Xd = {n: sp.hstack([T[n], sp.csr_matrix(sc.transform(Dd[n]))]).tocsr() for n in d}
    out['XGB+distress'] = xgb(42).fit(Xd['train'], y['train'], sample_weight=sw).predict(Xd['test'])
    pd.DataFrame(out).to_csv(R + f'{pre}_main_split_test_predictions.csv', index=False)
    for m in ('XGB', 'XGB+distress'):
        p, r_, f, s = precision_recall_fscore_support(y['test'], out[m], labels=[0, 1, 2], zero_division=0)
        print(f'{pre} {m}: ' + '; '.join(f'{c} P={p[i]:.3f} R={r_[i]:.3f} F1={f[i]:.3f} n={s[i]}' for i, c in enumerate(classes))
              + f' | macro-F1={f1_score(y["test"], out[m], average="macro"):.4f} acc={accuracy_score(y["test"], out[m]):.4f}')
main_split('bugzilla', ['Critical', 'High', 'Low'], 'bzmask', True)
main_split('github', ['Fast', 'Medium', 'Slow'], 'gh', False)
print('\nSaved all records in', R, ':', sorted(os.listdir(R)))
