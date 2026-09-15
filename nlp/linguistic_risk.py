"""Psycholinguistic risk scoring for consumer-complaint narratives.

This is the part of the system that is not a RAG wrapper. Instead of asking an
LLM "how angry does this sound", each narrative is scored with explicit,
inspectable features drawn from computational-psycholinguistics practice:

  hedging       epistemic hedges ("maybe", "I think", "sort of") — markers of low
                speaker commitment, which *dampen* an urgency read
  urgency       temporal-pressure and immediacy markers
  intensity     negative-affect lexicon, intensifiers, shouting (caps), "!"
  escalation    legal / regulatory threat vocabulary
  harm          concrete financial-consequence vocabulary
  repetition    evidence of repeated, unresolved contact attempts
  disfluency    low SLOR — see below

SLOR (Syntactic Log-Odds Ratio, Pauls & Klein 2012; Lau, Clark & Lappin 2017) is
the standard length- and frequency-controlled acceptability measure:

    SLOR(s) = ( log P_LM(s) - log P_unigram(s) ) / |s|

It subtracts the unigram probability of a sentence from its language-model
probability, so that a sentence is not penalised merely for containing rare
words, then normalises by length. Here P_LM comes from an add-k smoothed bigram
model fitted on the complaint corpus itself (see `NgramLM`) rather than a neural
LM — cheap, dependency-free, and reproducible on a free CPU box. Narratives with
markedly low SLOR are disfluent relative to the corpus norm, which in written
complaints tends to track distress and haste.

Every score is deterministic and every driver is traceable back to the exact
tokens that produced it, which is what makes the output defensible to a
compliance reviewer.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #
# Multi-word entries are matched as phrases; single words as whole tokens.

HEDGES = [
    "maybe", "perhaps", "possibly", "probably", "apparently", "seemingly",
    "somewhat", "arguably", "presumably", "supposedly", "allegedly",
    "i think", "i believe", "i guess", "i assume", "i suppose", "i feel like",
    "sort of", "kind of", "more or less", "as far as i know", "if i recall",
    "i am not sure", "im not sure", "not entirely sure", "it seems", "it appears",
    "might be", "may be", "could be", "should be able", "to some extent",
]

URGENCY = [
    "immediately", "immediate", "urgent", "urgently", "asap",
    "as soon as possible", "right away", "without delay", "time sensitive",
    "time-sensitive", "deadline", "overdue", "past due", "expires", "expiring",
    "running out", "cannot wait", "can not wait", "need this resolved",
    "still waiting", "still have not", "still havent", "no response",
    "never received", "has been weeks", "has been months", "months now",
    "weeks now", "days now", "every day", "today", "tonight", "tomorrow",
]

INTENSITY = [
    "furious", "outraged", "outrageous", "appalling", "appalled", "disgusted",
    "disgusting", "unacceptable", "unbelievable", "ridiculous", "absurd",
    "horrible", "horrific", "terrible", "awful", "nightmare", "devastating",
    "devastated", "humiliating", "humiliated", "insulting", "insulted",
    "fraud", "fraudulent", "scam", "scammed", "theft", "stolen", "stole",
    "robbed", "lied", "lying", "liar", "deceptive", "deceived", "misleading",
    "negligent", "negligence", "harassment", "harassed", "harassing",
    "discriminated", "discrimination", "predatory", "abusive", "abuse",
    "distress", "distressed", "desperate", "helpless", "trapped", "suffering",
    "angry", "frustrated", "frustrating", "frustration", "stressed", "anxiety",
    "crying", "panic", "panicking", "terrified", "scared", "afraid",
]

INTENSIFIERS = [
    "absolutely", "completely", "totally", "utterly", "extremely", "incredibly",
    "severely", "seriously", "highly", "deeply", "profoundly", "entirely",
    "beyond", "never ever", "at all", "whatsoever", "literally",
]

ESCALATION = [
    "attorney", "lawyer", "legal action", "legal counsel", "lawsuit", "sue",
    "suing", "litigation", "court", "small claims", "arbitration", "subpoena",
    "regulator", "regulatory", "attorney general", "class action",
    "better business bureau", "bbb", "occ", "fdic", "federal reserve",
    "state banking", "file a complaint", "filed a complaint", "report them",
    "reporting them", "escalate", "escalated", "escalation", "supervisor",
    "formal complaint", "fcra", "fdcpa", "reg e", "regulation e", "tila",
]

HARM = [
    "overdraft", "overdrawn", "nsf", "insufficient funds", "bounced",
    "returned check", "late fee", "late fees", "penalty", "penalties",
    "credit score", "credit report", "derogatory", "charge off", "charged off",
    "collections", "collection agency", "foreclosure", "foreclose", "eviction",
    "evicted", "repossession", "repossessed", "bankruptcy", "garnish",
    "garnishment", "lien", "frozen", "froze my", "locked out", "closed my account",
    "account closed", "denied", "denial", "declined", "rent", "mortgage",
    "payroll", "paycheck", "groceries", "medication", "homeless",
    "life savings", "retirement", "unable to pay", "cannot pay", "cant pay",
]

REPETITION = [
    "again and again", "over and over", "repeatedly", "multiple times",
    "several times", "numerous times", "countless times", "many times",
    "second time", "third time", "fourth time", "fifth time",
    "every time", "each time", "once again", "yet again", "still not",
    "still no", "keep calling", "kept calling", "called again", "no one",
    "nobody", "transferred", "runaround", "run around", "bounced around",
]

LEXICONS: dict[str, Sequence[str]] = {
    "hedging": HEDGES,
    "urgency": URGENCY,
    "intensity": INTENSITY,
    "intensifier": INTENSIFIERS,
    "escalation": ESCALATION,
    "harm": HARM,
    "repetition": REPETITION,
}

# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[a-z][a-z']*")
_WS_RE = re.compile(r"\s+")

# CFPB redacts personal details before publishing, leaving a small family of
# artefacts in the narrative text. Handled in order, most specific first.
_MONEY_RE = re.compile(r"\{\$([^}]*)\}")                       # {$1,234.56} → $1,234.56
_DATE_RE = re.compile(r"\bX{1,4}\s*/\s*X{1,4}\s*/\s*(?:X{2,4}|year>)", re.IGNORECASE)
_YEAR_TAG_RE = re.compile(r"\byear>", re.IGNORECASE)
_REDACTION_RE = re.compile(r"\bX{2,}\b")
_ORPHAN_SLASH_RE = re.compile(r"(?:\s/){2,}\s?")               # " / / " left by a stripped date


def _term_regex(term: str) -> str:
    """Whitespace in a phrase matches any run of whitespace; apostrophes optional.

    `"i'm not sure"` and `"im not sure"` are the same marker as far as the
    feature is concerned, and complaint text is inconsistent about both.
    """
    parts = [re.escape(word).replace("'", "'?") for word in term.split()]
    return r"\s+".join(parts)


def _compile_lexicon(terms: Sequence[str]) -> re.Pattern[str]:
    """One alternation regex per lexicon, longest-first so phrases win over words."""
    ordered = sorted(set(terms), key=len, reverse=True)
    alternation = "|".join(_term_regex(t) for t in ordered)
    return re.compile(rf"(?<![A-Za-z])(?:{alternation})(?![A-Za-z])", re.IGNORECASE)


_PATTERNS: dict[str, re.Pattern[str]] = {
    name: _compile_lexicon(terms) for name, terms in LEXICONS.items()
}


def normalize(text: str) -> str:
    """Strip CFPB redaction artefacts and collapse whitespace.

    Published narratives have names, dates and account numbers replaced with runs
    of `X`, dates written as `XX/XX/year>`, and amounts wrapped as `{$50.00}`.
    Left in place these distort the caps ratio (every run is uppercase) and the
    surprisal estimate (`xxxx` becomes the most frequent token in the corpus),
    and they make excerpts unreadable in the UI. Amounts are unwrapped and kept —
    a dollar figure is real evidence of harm; everything else is dropped.
    """
    if not text:
        return ""
    text = _MONEY_RE.sub(r"$\1", text)
    text = _DATE_RE.sub(" ", text)
    text = _YEAR_TAG_RE.sub(" ", text)
    text = _REDACTION_RE.sub(" ", text)
    text = _ORPHAN_SLASH_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Bigram LM for SLOR
# --------------------------------------------------------------------------- #

@dataclass
class NgramLM:
    """Add-k smoothed unigram+bigram model fitted on the complaint corpus.

    Small enough to ship inside the repo, which keeps the Hugging Face Space
    free of any build-time model download beyond the sentence-transformer.
    """

    unigrams: Counter = field(default_factory=Counter)
    bigrams: Counter = field(default_factory=Counter)
    total_tokens: int = 0
    vocab_size: int = 0
    k: float = 0.4

    BOS = "<s>"

    def fit(self, texts: Iterable[str], max_vocab: int = 50_000,
            max_bigrams: int = 400_000) -> "NgramLM":
        uni: Counter = Counter()
        bi: Counter = Counter()
        for text in texts:
            tokens = tokenize(normalize(text))
            if not tokens:
                continue
            uni.update(tokens)
            previous = self.BOS
            for token in tokens:
                bi[(previous, token)] += 1
                previous = token
        self.unigrams = Counter(dict(uni.most_common(max_vocab)))
        self.bigrams = Counter(dict(bi.most_common(max_bigrams)))
        self.total_tokens = sum(uni.values())
        self.vocab_size = max(len(self.unigrams), 1)
        return self

    # -- probabilities -------------------------------------------------------
    def log_p_unigram(self, token: str) -> float:
        count = self.unigrams.get(token, 0)
        return math.log((count + self.k) / (self.total_tokens + self.k * (self.vocab_size + 1)))

    def log_p_bigram(self, previous: str, token: str) -> float:
        numerator = self.bigrams.get((previous, token), 0) + self.k
        denominator = self.unigrams.get(previous, 0) + self.k * (self.vocab_size + 1)
        return math.log(numerator / denominator)

    def slor(self, text: str) -> float | None:
        """Syntactic log-odds ratio: length- and frequency-normalised fluency."""
        tokens = tokenize(normalize(text))
        if len(tokens) < 5 or not self.total_tokens:
            return None
        log_lm = 0.0
        log_uni = 0.0
        previous = self.BOS
        for token in tokens:
            log_lm += self.log_p_bigram(previous, token)
            log_uni += self.log_p_unigram(token)
            previous = token
        return (log_lm - log_uni) / len(tokens)

    def mean_surprisal(self, text: str) -> float | None:
        """Mean unigram surprisal in nats — the frequency term SLOR controls for."""
        tokens = tokenize(normalize(text))
        if not tokens:
            return None
        return -sum(self.log_p_unigram(t) for t in tokens) / len(tokens)

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "unigrams": dict(self.unigrams),
            "bigrams": {f"{a}\t{b}": c for (a, b), c in self.bigrams.items()},
            "total_tokens": self.total_tokens,
            "vocab_size": self.vocab_size,
            "k": self.k,
        }
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "NgramLM":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        model = cls(k=payload.get("k", 0.4))
        model.unigrams = Counter(payload["unigrams"])
        model.bigrams = Counter(
            {tuple(key.split("\t", 1)): count for key, count in payload["bigrams"].items()}
        )
        model.total_tokens = payload["total_tokens"]
        model.vocab_size = payload["vocab_size"]
        return model


# SLOR is corpus-relative, so raw values mean nothing on their own. These are the
# corpus mean/sd written by `data.ingest`; the defaults are a neutral fallback so
# the scorer still runs before an index has been built.
@dataclass
class SlorNorm:
    mean: float = 0.0
    sd: float = 1.0

    def z(self, value: float) -> float:
        return (value - self.mean) / (self.sd or 1.0)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

# Weights for the positive-risk features; they sum to 1.0. Hedging is applied
# separately as a dampener because low speaker commitment reduces, rather than
# adds to, an actionable urgency read.
WEIGHTS = {
    "urgency": 0.24,
    "intensity": 0.22,
    "escalation": 0.20,
    "harm": 0.18,
    "repetition": 0.10,
    "disfluency": 0.06,
}

# Saturation constants: the per-100-token rate at which a feature reaches ~63%
# of its maximum contribution. Tuned so a typical complaint lands mid-scale.
SATURATION = {
    "urgency": 1.6,
    "intensity": 1.8,
    "escalation": 0.9,
    "harm": 2.2,
    "repetition": 0.9,
    "hedging": 1.2,
}

HEDGE_DAMPENING = 0.25  # max proportion of the score heavy hedging can remove


@dataclass
class SeverityBands:
    """Cut points that turn a 0-100 score into low / medium / high.

    Absolute cut points do not survive contact with a real corpus: consumer
    complaints are long, so marker *rates* per 100 tokens are low and almost
    everything lands in the bottom band. Triage is a relative judgement anyway —
    "this is in the top decile of urgency for this book of complaints" is the
    statement an analyst can act on — so `data.ingest` calibrates these against
    the corpus and stores them in index_meta.json. The defaults below are only a
    fallback for scoring ad-hoc text before an index exists.
    """

    medium: float = 35.0
    high: float = 60.0
    # The 99 inner percentile cut points of the corpus score distribution, used to
    # express a raw score as a rank. Raw scores are compressed — a narrative would
    # have to max out every feature at once to approach 100 — so "34/100" and
    # "high" look contradictory until the rank is shown next to them.
    cuts: list[float] = field(default_factory=list)

    def label(self, score: float) -> str:
        if score >= self.high:
            return "high"
        if score >= self.medium:
            return "medium"
        return "low"

    def percentile(self, score: float) -> int | None:
        """Where this score sits in the corpus, 0-100."""
        if not self.cuts:
            return None
        from bisect import bisect_right

        return min(100, bisect_right(self.cuts, score))

    @classmethod
    def from_scores(cls, scores: Sequence[float],
                    medium_pct: int = 70, high_pct: int = 92) -> "SeverityBands":
        """Calibrate so ~30% of the corpus is medium-or-worse and ~8% is high."""
        if not scores:
            return cls()
        import statistics as _stats

        cuts = _stats.quantiles(sorted(scores), n=100, method="inclusive")
        return cls(
            medium=round(cuts[medium_pct - 1], 1),
            high=round(cuts[high_pct - 1], 1),
            cuts=[round(c, 3) for c in cuts],
        )


def _saturate(rate: float, constant: float) -> float:
    """Map an unbounded per-100-token rate onto [0, 1) with diminishing returns."""
    return 1.0 - math.exp(-rate / constant)


@dataclass
class RiskScore:
    """The full, inspectable output for one narrative."""

    risk_score: float           # 0-100, raw model output
    severity: str               # low | medium | high
    percentile: int | None      # rank within the indexed corpus
    n_tokens: int
    features: dict[str, float]  # saturated [0,1] feature strengths
    rates: dict[str, float]     # raw hits per 100 tokens
    matches: dict[str, list[str]]
    slor: float | None
    slor_z: float | None
    mean_surprisal: float | None
    drivers: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def headline(self) -> str:
        rank = f", {ordinal(self.percentile)} pct" if self.percentile is not None else ""
        return f"{self.severity.upper()} (raw {self.risk_score:.0f}/100{rank})"

    def summary(self) -> str:
        driver_text = ", ".join(self.drivers) if self.drivers else "no strong markers"
        return f"{self.headline()} — {driver_text}"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class LinguisticRiskScorer:
    """Scores narratives. Stateless apart from the optional corpus LM."""

    def __init__(self, lm: NgramLM | None = None, slor_norm: SlorNorm | None = None,
                 bands: SeverityBands | None = None):
        self.lm = lm
        self.slor_norm = slor_norm or SlorNorm()
        self.bands = bands or SeverityBands()

    # -- individual feature extraction ---------------------------------------
    @staticmethod
    def _lexicon_hits(text: str, name: str) -> list[str]:
        return [m.group(0).lower() for m in _PATTERNS[name].finditer(text)]

    @staticmethod
    def _caps_ratio(text: str) -> float:
        """Share of alphabetic words written in all-caps (shouting), 4+ letters.

        The length floor keeps acronyms like ATM, APR and PIN from reading as
        shouting; CFPB narratives are full of them.
        """
        words = re.findall(r"[A-Za-z]{4,}", text)
        if not words:
            return 0.0
        shouted = sum(1 for w in words if w.isupper())
        return shouted / len(words)

    def score(self, text: str) -> RiskScore:
        clean = normalize(text)
        tokens = tokenize(clean)
        n_tokens = len(tokens)
        if n_tokens == 0:
            return RiskScore(0.0, "low", None, 0, {}, {}, {}, None, None, None, [])

        per100 = 100.0 / n_tokens
        matches: dict[str, list[str]] = {}
        rates: dict[str, float] = {}

        for name in ("hedging", "urgency", "escalation", "harm", "repetition"):
            hits = self._lexicon_hits(clean, name)
            matches[name] = hits
            rates[name] = len(hits) * per100

        # Intensity blends its lexicon with intensifiers, shouting and "!".
        affect_hits = self._lexicon_hits(clean, "intensity")
        intensifier_hits = self._lexicon_hits(clean, "intensifier")
        caps_ratio = self._caps_ratio(clean)
        exclaim_rate = clean.count("!") * per100
        matches["intensity"] = affect_hits + intensifier_hits
        rates["intensity"] = (
            len(affect_hits) * per100
            + 0.5 * len(intensifier_hits) * per100
            + 6.0 * caps_ratio
            + 0.5 * exclaim_rate
        )

        features = {
            name: _saturate(rates[name], SATURATION[name])
            for name in ("urgency", "intensity", "escalation", "harm", "repetition", "hedging")
        }

        # Disfluency: how far below the corpus mean this narrative's SLOR sits.
        slor = self.lm.slor(clean) if self.lm else None
        mean_surprisal = self.lm.mean_surprisal(clean) if self.lm else None
        slor_z = self.slor_norm.z(slor) if slor is not None else None
        if slor_z is None:
            features["disfluency"] = 0.0
        else:
            features["disfluency"] = min(1.0, max(0.0, -slor_z / 2.0))

        raw = sum(WEIGHTS[name] * features[name] for name in WEIGHTS)
        dampener = 1.0 - HEDGE_DAMPENING * features["hedging"]
        risk = round(100.0 * raw * dampener, 1)

        severity = self.bands.label(risk)

        contributions = {name: WEIGHTS[name] * features[name] for name in WEIGHTS}
        drivers = [
            self._describe(name, matches.get(name, []))
            for name, value in sorted(contributions.items(), key=lambda kv: -kv[1])[:3]
            if value > 0.02
        ]
        if features["hedging"] > 0.4:
            drivers.append("heavily hedged (dampens urgency)")

        rates["caps_ratio"] = caps_ratio
        rates["exclaim_per_100"] = exclaim_rate

        return RiskScore(
            risk_score=risk,
            severity=severity,
            percentile=self.bands.percentile(risk),
            n_tokens=n_tokens,
            features={k: round(v, 3) for k, v in features.items()},
            rates={k: round(v, 3) for k, v in rates.items()},
            matches={k: sorted(set(v))[:8] for k, v in matches.items() if v},
            slor=round(slor, 4) if slor is not None else None,
            slor_z=round(slor_z, 2) if slor_z is not None else None,
            mean_surprisal=round(mean_surprisal, 3) if mean_surprisal is not None else None,
            drivers=drivers,
        )

    @staticmethod
    def _describe(name: str, hits: list[str]) -> str:
        label = {
            "urgency": "urgency markers",
            "intensity": "emotional intensity",
            "escalation": "escalation/legal language",
            "harm": "concrete financial harm",
            "repetition": "repeated unresolved contact",
            "disfluency": "disfluent phrasing (low SLOR)",
        }[name]
        if not hits:
            return label
        examples = ", ".join(sorted(set(hits))[:3])
        return f"{label} ({examples})"

    def score_many(self, texts: Iterable[str]) -> list[RiskScore]:
        return [self.score(t) for t in texts]


def load_scorer(lm_path: str | Path | None = None, norm: SlorNorm | None = None,
                bands: SeverityBands | None = None) -> LinguisticRiskScorer:
    """Build a scorer, attaching the corpus LM if one has been fitted."""
    lm = None
    if lm_path and Path(lm_path).exists():
        lm = NgramLM.load(lm_path)
    return LinguisticRiskScorer(lm=lm, slor_norm=norm, bands=bands)


if __name__ == "__main__":  # quick manual check
    scorer = LinguisticRiskScorer()
    samples = [
        "I have called Wells Fargo five times about an overdraft fee that was "
        "charged after my deposit cleared. Nobody will help me. I am now unable "
        "to pay my rent and my credit score is dropping. This is absolutely "
        "unacceptable and I am contacting an attorney immediately.",
        "I think there may possibly have been a small error on my statement. "
        "It could be my own misreading, but I would appreciate someone taking a "
        "look at it when they get a chance. Thank you for your time.",
    ]
    for sample in samples:
        print(scorer.score(sample).summary())
