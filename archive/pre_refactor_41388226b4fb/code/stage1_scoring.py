"""Deterministic scoring rules for both tasks. No LLM judges.

Task 1 (naming): correct / abstain / certain-wrong, using LVIS name+synonyms
plus WordNet DIRECT hypernym lemmas of the category synset.
Task 2 (existence): yes/no from the generated text.
"""
import os, re, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('NLTK_DATA', str(ROOT / 'cache' / 'nltk'))
import nltk
from nltk.corpus import wordnet as wn

_ARTICLES = r'^(a|an|the)\s+'
_cached_sets = {}


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\([^)]*\)', ' ', s)   # drop LVIS parenthetical glosses
    s = re.sub(r'[.,;:!?"()\[\]]', ' ', s)
    s = re.sub(_ARTICLES, '', s)
    s = s.replace('-', '_').replace(' ', '_')
    s = re.sub(r'_+', '_', s).strip('_')
    return s


def accept_set(name, synonyms, synset_str):
    """Name + synonyms + WordNet direct hypernym lemmas, all normalized."""
    key = (name, synset_str)
    if key in _cached_sets:
        return _cached_sets[key]
    acc = {_norm(name)}
    for s in synonyms or []:
        acc.add(_norm(s))
    try:
        if synset_str and synset_str.count('.') == 2:
            ss = wn.synset(synset_str)
            for hyp in ss.hypernyms():          # direct hypernyms only
                for lemma in hyp.lemma_names():
                    acc.add(_norm(lemma))
            for lemma in ss.lemma_names():
                acc.add(_norm(lemma))
    except Exception:
        pass
    acc.discard('')
    _cached_sets[key] = acc
    return acc


ABSTAIN_MARKERS = ('unknown', "i_don't_know", 'not_sure', 'cannot_identify',
                   'cannot_recognize', 'unsure', 'no_idea', 'cant_identify',
                   'unable_to_identify', 'unable_to_determine', 'not_recognizable')


def is_abstain(text: str) -> bool:
    n = _norm(text)
    if not n:
        return True  # empty output = no committed answer; count as abstain
    for m in ABSTAIN_MARKERS:
        if n == m or n.startswith(m + '_'):
            return True
    return False


def _sing(w: str) -> str:
    """Naive English singularization for plural tolerance."""
    if len(w) > 4 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) > 3 and w.endswith('es') and not w.endswith('ses'):
        return w[:-2]
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
        return w[:-1]
    return w


def score_naming(text: str, name, synonyms, synset_str):
    """Returns one of: 'correct', 'abstain', 'wrong' (deterministic error)."""
    if is_abstain(text):
        return 'abstain'
    n = _norm(text)
    acc = accept_set(name, synonyms, synset_str)
    # candidate spans: full string, each 1-3-gram, plural-insensitive
    words = n.split('_')
    cands = {n, '_'.join(_sing(w) for w in words)}
    for k in (1, 2, 3):
        for i in range(len(words) - k + 1):
            gram = '_'.join(words[i:i + k])
            cands.add(gram)
            cands.add('_'.join(_sing(w) for w in gram.split('_')))
    if cands & acc:
        return 'correct'
    return 'wrong'


def score_existence(text: str, gold: bool):
    """gold=True means the category IS present. Returns 'correct'/'wrong'/'wrong_unparsed'."""
    n = _norm(text)
    yes = bool(re.match(r'^(yes|yeah|yep|true|it_is|there_is)', n))
    no = bool(re.match(r'^(no|nope|false|not|there_is_no|i_don)', n)) or n.startswith('no_')
    if yes and not no:
        pred = True
    elif no:
        pred = False
    else:
        return 'wrong_unparsed'
    return 'correct' if pred == gold else 'wrong'


# ---------------- metric definitions (report section 五) ----------------
def metrics(rows):
    """rows: list of dicts with 'outcome' in {correct, abstain, wrong} (naming)
    or {correct, wrong} (existence; abstain impossible by design).
    Returns accuracy / abstain_rate / certain_error_rate."""
    n = len(rows)
    if n == 0:
        return {'n': 0}
    acc = sum(r['outcome'] == 'correct' for r in rows) / n
    abst = sum(r['outcome'] == 'abstain' for r in rows) / n
    wrong = sum(r['outcome'] in ('wrong', 'wrong_unparsed') for r in rows) / n
    return {'n': n, 'accuracy': acc, 'abstain_rate': abst, 'certain_error_rate': wrong}


def calibration_error(confs, corrects, n_bins=10):
    """Weighted binned ECE: sum_b (n_b/N) * |mean_conf_b - acc_b|.
    confs: max probability at the answer position; corrects: 0/1."""
    n = len(confs)
    if n == 0:
        return float('nan')
    edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for i in range(n_bins):
        idx = [j for j in range(n) if (edges[i] <= confs[j] < edges[i + 1]) or
               (i == n_bins - 1 and confs[j] == 1.0)]
        if idx:
            mc = sum(confs[j] for j in idx) / len(idx)
            ac = sum(corrects[j] for j in idx) / len(idx)
            ece += len(idx) / n * abs(mc - ac)
    return ece
