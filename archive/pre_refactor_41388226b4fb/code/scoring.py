"""Deterministic scoring for v2 (food101-style fine-grained classes).

Naming: correct / abstain / wrong, accepting the official class name, WordNet
lemma variants, and DIRECT hypernym lemmas — except hypernyms that are shared by
more than 30% of all classes with synsets (genericity guard, so that answering
"food"/"dish" is not counted correct).
Style-2 outputs carry a RECOGNIZED/NOT_RECOGNIZED prefix; style-3 abstention is
decided by the round-A YES/NO answer (handled by the caller).
"""
import os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('NLTK_DATA', str(ROOT / 'cache' / 'nltk'))
import nltk
from nltk.corpus import wordnet as wn

_ARTICLES = r'^(a|an|the)\s+'
_accept_cache = {}
_generic_set = None


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = re.sub(r'[.,;:!?"()\[\]]', ' ', s)
    s = re.sub(_ARTICLES, '', s)
    s = s.replace('-', '_').replace(' ', '_')
    s = re.sub(r'_+', '_', s).strip('_')
    return s


def _sing(w: str) -> str:
    if len(w) > 4 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) > 3 and w.endswith('es') and not w.endswith('ses'):
        return w[:-2]
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
        return w[:-1]
    return w


def class_synsets(class_name):
    """Best-effort WordNet synsets for a dataset class name like 'beef_tartare'."""
    phrase = class_name.replace('_', ' ')
    cands = []
    for ss in wn.synsets(phrase, pos=wn.NOUN):
        cands.append(ss)
    if not cands and '_' in class_name:
        for part in phrase.split():
            for ss in wn.synsets(part, pos=wn.NOUN):
                cands.append(ss)
    return cands


def generic_hypernyms(all_classes, thresh=0.30):
    """Hypernym lemmas that cover > thresh of classes (e.g. food/dish/plate) —
    accepting them would make almost any answer correct."""
    global _generic_set
    if _generic_set is not None:
        return _generic_set
    cov = {}
    n_with = 0
    for c in all_classes:
        hyps = set()
        for ss in class_synsets(c):
            for h in ss.hypernyms():
                for l in h.lemma_names():
                    hyps.add(_norm(l))
        if hyps:
            n_with += 1
        for h in hyps:
            cov[h] = cov.get(h, 0) + 1
    _generic_set = {h for h, k in cov.items() if n_with and k / n_with > thresh}
    return _generic_set


def accept_set(class_name, all_classes):
    key = class_name
    if key in _accept_cache:
        return _accept_cache[key]
    acc = {_norm(class_name)}
    gen = generic_hypernyms(all_classes)
    for ss in class_synsets(class_name):
        for lemma in ss.lemma_names():
            n = _norm(lemma)
            if n not in gen:
                acc.add(n)
        for hyp in ss.hypernyms():  # direct hypernyms only
            for lemma in hyp.lemma_names():
                n = _norm(lemma)
                if n not in gen:
                    acc.add(n)
    acc.discard('')
    _accept_cache[key] = acc
    return acc


ABSTAIN_MARKERS = ('unknown', "i_don't_know", 'not_sure', 'cannot_identify',
                   'cannot_recognize', 'unsure', 'no_idea', 'cant_identify',
                   'unable_to_identify', 'unable_to_determine', 'not_recognizable',
                   'not_recognized', 'cannot_determine', 'cannot_recognise',
                   'unrecognized', 'unidentifiable', 'not_familiar', 'cant_recognize')


def is_abstain(text: str) -> bool:
    n = _norm(text)
    if not n:
        return True
    for m in ABSTAIN_MARKERS:
        if n == m or n.startswith(m + '_') or n.endswith('_' + m):
            return True
    return False


def score_naming(text: str, class_name, all_classes):
    """Returns 'correct' | 'abstain' | 'wrong'."""
    if is_abstain(text):
        return 'abstain'
    n = _norm(text)
    acc = accept_set(class_name, all_classes)
    words = n.split('_')
    cands = {n, '_'.join(_sing(w) for w in words)}
    for k in (1, 2, 3, 4):
        for i in range(len(words) - k + 1):
            gram = '_'.join(words[i:i + k])
            cands.add(gram)
            cands.add('_'.join(_sing(w) for w in gram.split('_')))
    if cands & acc:
        return 'correct'
    return 'wrong'


def parse_style2(text: str):
    """Returns (abstain: bool, name_text: str)."""
    n = _norm(text)
    if is_abstain(text):
        return True, ''
    if 'not_recognized' in n or 'not_recognisable' in n:
        return True, ''
    if 'recognized' in n or 'recognised' in n:
        # strip the marker and take the remainder as the name
        rest = re.sub(r'^(re)?cognized[,_ ]*', '', n.replace('_', ' '))
        return False, rest
    return False, text


def score_existence(text: str, gold: bool):
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


def metrics(rows):
    n = len(rows)
    if n == 0:
        return {'n': 0}
    return {'n': n,
            'accuracy': sum(r['outcome'] == 'correct' for r in rows) / n,
            'abstain_rate': sum(r['outcome'] == 'abstain' for r in rows) / n,
            'certain_error_rate': sum(r['outcome'] in ('wrong', 'wrong_unparsed') for r in rows) / n}


def wrong_answer_confidence(rows):
    """Mean first-token max prob over certain-error answers (core outcome v2)."""
    confs = [r.get('answer_maxp') for r in rows
             if r['outcome'] in ('wrong', 'wrong_unparsed') and r.get('answer_maxp') is not None]
    if not confs:
        return float('nan'), 0
    return sum(confs) / len(confs), len(confs)


def calibration_error(confs, corrects, n_bins=10):
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
