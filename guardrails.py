"""
guardrails.py -- word pools, usage ledger, and prompt construction.

The guardrail is three things, and only one of them is a prompt:

  1. POOL    (data)  -- POS-routed. A verb pattern draws from verbs, never nouns.
  2. LEDGER  (state) -- persistent per-(word, grammar point) usage counts, so the
                        next call draws the *least used* words. This is what turns
                        coverage from a hope into a construction.
  3. CHECK   (later) -- validate.py enforces required_word_present after the fact,
                        and generate.py retries with the failure stated. The prompt
                        raises the hit rate; the retry guarantees it.

Measured motivation: with an unrotated alphabetical hint the model produced 69%
vocabulary coverage and repeated 水を飲む 6 times. Prompts alone do not enforce.
"""

import json, os, re, random, collections

OUT = r"H:\jlpt\gen"

# --- layout -------------------------------------------------------------
# OUT  = this repo: the pipeline code
# DATA = curated inputs the code reads (vocabulary, targets, classes)
# RUNS = every generated artefact (attempts, sentences, logs, ledgers)
DATA = os.path.join(OUT, "data")
RUNS = r"H:\jlpt\runs"

# POS classes as they appear in lexicon.json's n5_vocab[].pos
POOL_POS = ("verb", "noun", "i-adj", "na-adj", "adverb")


def build_pools(lex):
    """POS-routed pools from the N5 vocabulary."""
    pools = collections.defaultdict(list)
    for v in lex["n5_vocab"]:
        pools[v.get("pos", "other")].append(v)
    # adjective patterns accept either subclass
    pools["adj"] = pools["i-adj"] + pools["na-adj"]
    return pools


POS_OVERRIDE = {
    # "Plain or polite form + から" is a verb frame, but the guardrail word must be
    # the REASON's subject matter, not an arbitrary verb. Handing it a bare verb
    # produced 「忘れるから、書きます」 (because I'll forget, I write),
    # 「ボタンを押すから、見てください」 -- the required word was forced into the
    # reason slot and the causality collapsed. A noun gives a natural reason:
    # 「お金がないから、買いません」. (paired with a `frame` on the target)
    "kara-cause": "noun",
}


def pos_need(pattern, meaning_en="", gp=None):
    """Which POS a grammar point's pattern demands.

    The FIRST POS the pattern mentions wins. Patterns list alternatives
    ('V plain / i-adj / na-adj な / noun の + とき(に)') and the first one is the
    primary frame; picking any other hands the model a pool it cannot easily use.
    This also makes the '(+ verb)' companion harmless, since a companion is never
    mentioned first.

    `gp` is optional so existing callers keep working; when given, a per-point
    override can name a POS the pattern text would not imply.
    """
    if gp and gp in POS_OVERRIDE:
        return POS_OVERRIDE[gp]
    p = f"{pattern or ''} {meaning_en or ''}".lower()
    order = [
        (r"\bv\b|\bverb|動詞|plain form|polite form", "verb"),
        (r"i-adj|い-adjective|i adjective", "i-adj"),
        (r"na-adj|な-adjective|na adjective", "na-adj"),
        (r"adjective|形容詞", "adj"),
        (r"noun|名詞|number|counter", "noun"),
    ]
    best, best_at = None, 10 ** 9
    for rx, pos in order:
        m = re.search(rx, p)
        if m and m.start() < best_at:
            best, best_at = pos, m.start()
    return best or "noun"          # counters attach to nouns
    if "noun" in p:
        return "noun"
    # patterns that mention no POS: they are mostly copula/particle frames,
    # which take a noun subject
    return "noun"


# ---- semantic classes -------------------------------------------------------
# POS alone is not enough. Requiring a NOUN for counter-en-money produced
#   お姉さんは五百円です (my sister is 500 yen)    -- PERSON in a price slot
#   グラムが百あります (there are 100 grams)       -- a UNIT counted as an object
# お姉さん and グラム are both nouns; nothing but meaning distinguishes them.
# noun_classes.json (built by classify_nouns.py) supplies the meaning.

_NC_PATH = os.path.join(DATA, "noun_classes.json")
_NCO_PATH = os.path.join(DATA, "noun_class_overrides.json")

_classes_cache = None


def load_classes(path=None):
    """Semantic classes for N5 nouns, with hand overrides applied on top.

    Overrides are merged here rather than baked into noun_classes.json so a
    correction takes effect immediately without re-running classification.
    """
    global _classes_cache
    if path is None and _classes_cache is not None:
        return _classes_cache

    data = {}
    p = path or _NC_PATH
    if os.path.exists(p):
        try:
            data = json.load(open(p, encoding="utf-8"))
        except Exception:
            data = {}

    if path is None and os.path.exists(_NCO_PATH):
        try:
            ov = json.load(open(_NCO_PATH, encoding="utf-8"))
        except Exception:
            ov = {}
        for w, c in ov.items():
            if w.startswith("_"):        # _comment etc
                continue
            data.setdefault(w, {}).update(c)

    if path is None:
        _classes_cache = data
    return data


# Lexicon part-of-speech detail from JMdict (pos_raw), keyed by word. Independent of
# noun_classes: noun_classes only covers nouns the classifier was asked about, while
# pos_raw exists for every vocabulary entry and carries transitivity (vt/vi), which
# is the one thing a regex cannot recover from the surface string.
_POS_RAW = None


def _pos_raw():
    global _POS_RAW
    if _POS_RAW is None:
        try:
            lex = json.load(open(os.path.join(DATA, "lexicon.json"), encoding="utf-8"))
            _POS_RAW = {v["text"]: set(v.get("pos_raw") or ()) for v in lex.get("n5_vocab", [])}
        except Exception:
            _POS_RAW = {}
    return _POS_RAW


# A noun passes a rule if it matches ANY listed axis (union), because a noun can
# qualify either by what it is (kind) or by how it is counted (shape).
CLASS_RULES = {
    # -- counters: the required noun IS the counted noun --------------------
    "counter-en-money":   {"kind": {"FOOD", "OBJECT", "CLOTHING", "VEHICLE",
                                    "EVENT", "NATURE", "SUBSTANCE"}},
    "counter-mai-flat":   {"shape": {"FLAT"}},
    "counter-hon-long":   {"shape": {"LONG"}},
    "counter-people-nin": {"kind": {"PERSON"}},
    "counter-sai-age":    {"kind": {"PERSON"}},
    "counter-tsu":        {"kind": {"OBJECT", "FOOD", "NATURE"},
                           "shape": {"ROUND", "CONTAINER"}},
    # -- counters where the noun is the OBJECT OF AN ACTION -----------------
    # (the pattern is spent as "X を N時/N分 に する")
    # -- time counters: the noun is a TIME REFERENCE, not an object ---------
    # The old OBJECT/FOOD/... rule here produced 切手/箸/ペン, which the model
    # dutifully spent as 「切手を買うのは三時です」 (buying stamps is three
    # o'clock). TIME alone is still too broad -- it carries weekdays, years,
    # months and adverbs, none of which can head a clock sentence (「火曜日は
    # 三時です」/「来年は三時です」), and a deny-list for those kept growing. So
    # this is an allow-list of words that actually work in "X は N時です".
    "counter-ji-oclock":  {"text_match":
        r"^(今|朝|今朝|明日|今日|昨日|一昨日|午前|午後|夕方|夜|今晩|晩|昼|"
        r"毎朝|毎晩|毎日|昨夜)$"},
    "counter-fun-minute": {"text_match":
        r"^(今|朝|今朝|明日|今日|昨日|一昨日|午前|午後|夕方|夜|今晩|晩|昼|"
        r"毎朝|毎晩|毎日|昨夜)$"},
    # Number + 時間: previously had NO rule, so the pool was all 381 nouns --
    # which is how 伯母さん/先生/本棚 reached a duration pattern.
    "jikan-time-duration": {"text_match":
        r"^(今|朝|今朝|明日|今日|昨日|午前|午後|夕方|夜|今晩|毎朝|毎晩|毎日)$"},
    # location relatives: no semantic class covers 上/下/中/前/後ろ/横/隣/間 --
    # they classify as ABSTRACT, PLACE, or nothing at all -- so match the words.
    # The earlier rule also allow-listed 右/左/外/内/向こう/近く/そば/あいだ. Those
    # are position words but they CANNOT take the "Noun + の + X" form the pattern
    # requires, so they produced 「右に車があります」/「外に犬がいます」 -- 51% of
    # this point's sentences missed the pattern and the pool was the cause.
    "location-nouns-ue-shita": {
        "text_match": r"^(上|下|中|前|後ろ|横|隣|間)$"},
    # な-adjective に-adverbial. An ALLOW-list, because of the 29 na-adj in the
    # lexicon many are adverbs or nouns smuggled in (あまり/たくさん/多分/いかが/
    # いろいろ/本当/まだ/まっすぐ) and several simply do not take 〜に+verb
    # (結構/同じ/黄色/大人/有名/嫌い). Those produced 「結構に使います」/「同じに作ります」.
    "na-adj-adverbial-ni": {
        "text_match": r"^(上手|下手|大切|静か|元気|きれい|大変|丈夫|立派|賑やか|便利|親切|丁寧|簡単)$"},
    # ませんか is an INVITATION. State and ability verbs make plain yes/no questions
    # instead: 「水が要りませんか」/「日本語が出来ませんか」/「駅に着きませんか」 are
    # questions about a state, never offers.
    "masenka-invitation": {
        "not_text": r"^(要る|出来る|有る|居る|分かる|見える|聞こえる|着く|違う|成る|似合う|足りる)$"},
    # 「を」 must mark the DIRECT OBJECT. 「〜をする」 set phrases put を inside a fixed
    # expression rather than on the acted-upon noun: 「日曜日に買い物をします」.
    # (The lexicon tags those nouns as verbs, so they reach a verb-routed pool.)
    # を must mark a DIRECT OBJECT, so the pool is restricted to TRANSITIVE verbs via
    # the lexicon's JMdict transitivity. 60 of the point's 136 verbs are vi-only, and
    # an intransitive cannot take an object -- を ends up marking a path or departure
    # (「公園を散歩します」/「バスを降ります」/「道を走ります」). That single mistake
    # caused both of this point's symptoms: off-pattern sentences (which the
    # per-sentence filter then dropped) AND the coverage collapse that followed.
    # The short deny-list stays for サ変 set phrases whose を is a fixed collocation
    # (「洗濯をします」) even though 洗濯 is itself transitive.
    "particle-o-object": {
        "pos_raw_any": {"vt"},
        "not_text": r"^(買い物|洗濯|掃除|勉強|電話|仕事|食事|散歩|旅行|料理)$"},
    # この/その/あの/どの + Noun needs a DEMONSTRABLE thing. The bare 381-noun pool
    # admits counter words (八つ is ABSTRACT) and time words, giving
    # 「この本は八つあります」/「その日から二日です」.
    "kono-sono-ano-dono-attributive": {
        "kind": {"OBJECT", "FOOD", "NATURE", "PLACE", "PERSON", "ANIMAL",
                 "VEHICLE", "CLOTHING", "SUBSTANCE"},
        "not_kind": {"ABSTRACT", "TIME", "UNIT", "BODY"}},
    # Noun + でした wants the NOUN as the predicate. The full 381-noun pool let the
    # model spend the required noun as the SUBJECT and a na-adjective as the
    # predicate -- 「村は静かでした」 -- which passes word coverage and matches
    # /でした/ while using no noun-copula at all.
    "deshita-past-polite-copula": {
        "kind": {"TIME", "PLACE", "PERSON", "OBJECT", "FOOD", "NATURE", "EVENT"},
        "not_kind": {"ABSTRACT", "UNIT"}},
    "ikura-how-much":     {"kind": {"FOOD", "OBJECT", "CLOTHING", "VEHICLE",
                                    "EVENT"}},
    # いくつ counts things. NATURE admitted 緑 (not countable) and the shape
    # axis admitted 人/二人 (PERSON), so both are excluded.
    "ikutsu-how-many":    {"kind": {"OBJECT", "FOOD"},
                           "shape": {"ROUND", "CONTAINER"},
                           "not_kind": {"PERSON", "ANIMAL", "UNIT", "TIME",
                                        "PLACE", "ABSTRACT", "BODY", "EVENT"}},
    # a な-adjective must work AS one in 「〜ではありません」. The lexicon's na-adj
    # list also carries adverbs (あまり, たくさん, 多分, いかが, いろいろ, 大変,
    # まだ, まっすぐ) and nouns (大人, 黄色, 本当); 結構 and 同じ are な-adjective
    # in form but not in this frame.
    "na-adjective-negative": {
        "not_text": r"^(あまり|いかが|いろいろ|結構|たくさん|多分|大変|黄色|大人|"
                    r"同じ|本当|まっすぐ|まだ)$"},
    # -- existence: あります takes things, います takes people and animals ---
    "arimasu-existence-inanimate": {
        "kind": {"OBJECT", "FOOD", "NATURE", "SUBSTANCE", "VEHICLE"},
        "shape": {"FLAT", "LONG", "ROUND", "CONTAINER", "MACHINE"}},
    "imasu-existence-animate": {"kind": {"PERSON", "ANIMAL"}},
    # 毎日/毎年/毎晩 are all TIME but none is a weekday, so a semantic class is
    # not enough here -- the only shared property is the word 曜日 itself.
    "nanyoubi-day-of-week": {"text_match": r"曜日"},
}


def noun_ok(gp, word, classes=None):
    """Does `word` suit grammar point `gp`?

    Unknown grammar points and unclassified words both pass -- we only filter
    where we have positive knowledge that the word fits.

    A rule may use any of:
      text_match -- a regex on the word itself. Some content requirements are not
                    a semantic class: the days of the week are ordinary TIME
                    nouns, and the only thing they have in common is 曜日.
      kind/shape -- semantic classes from noun_classes.json.
      not_text   -- exclusion regex. Needed because the positive axes are a UNION:
                    ikutsu-how-many's shape:{ROUND} admitted 人 and 二人 (PERSON),
                    so the only way to bar a word is to bar it explicitly.
      not_kind   -- exclusion classes, same reason.
      strict     -- unclassified words do NOT pass. A allow-list rule ("time words
                    only") must reject them; the default is to let them through.
    """
    rule = CLASS_RULES.get(gp)
    if not rule:
        return True

    # exclusions run before any positive test
    nt = rule.get("not_text")
    if nt and re.search(nt, word):
        return False

    # Lexicon transitivity (JMdict vt/vi), independent of noun_classes so it also
    # applies to words the noun classifier never saw. This is what fixes
    # particle-o-object: 60 of its 136 pool verbs are vi-only, and an intransitive
    # verb cannot take a direct object, so を ends up marking a PATH or a DEPARTURE
    # (「公園を散歩します」/「バスを降ります」/「道を走ります」) rather than an object.
    # Both symptoms came from that one mistake: off-pattern sentences the filter then
    # dropped, and coverage that collapsed with them.
    pr_any = rule.get("pos_raw_any")
    pr_none = rule.get("pos_raw_none")
    if pr_any or pr_none:
        pr = _pos_raw().get(word, set())
        if pr_any and not (pr & set(pr_any)):
            return False
        if pr_none and (pr & set(pr_none)):
            return False

    tm = rule.get("text_match")
    if tm:
        return bool(re.search(tm, word))

    classes = classes if classes is not None else load_classes()
    c = classes.get(word)
    if not c:
        return not rule.get("strict")

    nk = set(rule.get("not_kind") or ())
    if set(c.get("kind") or ()) & nk:
        return False
    if set(c.get("shape") or ()) & nk:
        return False

    for axis, allowed in rule.items():
        if axis in ("text_match", "not_text", "not_kind", "strict"):
            continue
        if set(c.get(axis) or ()) & set(allowed):
            return True
    return False


def filter_pool(gp, pool, classes=None):
    """Pool reduced to words that suit this grammar point.

    Falls back to the unfiltered pool if the rule would empty it, so a bad rule
    degrades to previous behaviour instead of producing no sentences.
    """
    if gp not in CLASS_RULES:
        return pool
    classes = classes if classes is not None else load_classes()
    kept = [v for v in pool if noun_ok(gp, v["text"], classes)]
    return kept if kept else pool


# ---- orthography: kana where N5 expects kanji -------------------------------
# Two reasons this matters beyond looks:
#   1. the corpus should not teach kana for a word N5 writes in kanji;
#   2. janome mis-segments kana-written words -- はこ becomes は + こ -- so
#      content_lemmas gains a meaningless 'こ' and loses '箱'. A learner typing
#      箱 then fails to match a corpus entry stored as はこ. That is key corruption,
#      not cosmetics.
#
# A general detector is NOT achievable. Substring matching over readings flags 84%
# of the corpus (は is both a particle and the reading of 歯); token-reading matching
# misses はこ because the tokeniser splits it; joining kana runs reintroduces the
# first problem. Hence a curated list, plus a guard that rejects matches crossing a
# word boundary (は + これ must not read as はこ).

CURATED_KANA = {
    "ほしい": "欲しい", "はこ": "箱", "つくえ": "机", "くるま": "車", "くつ": "靴",
    "あさ": "朝", "はな": "花", "げんき": "元気", "おふろ": "お風呂",
    "おさけ": "お酒", "むずかしい": "難しい", "あたたかい": "暖かい",
    "つめたい": "冷たい", "たのしい": "楽しい", "せんせい": "先生",
    "がっこう": "学校", "ともだち": "友達", "おかね": "お金", "たまご": "卵",
    "さかな": "魚", "やさい": "野菜", "くだもの": "果物", "ようふく": "洋服",
    "でんしゃ": "電車", "じてんしゃ": "自転車", "びょういん": "病院",
    "ぎんこう": "銀行", "ゆうびんきょく": "郵便局", "しんぶん": "新聞",
    "じしょ": "辞書", "でんわ": "電話", "こども": "子供", "おとこのこ": "男の子",
    "おんなのこ": "女の子", "おとこのひと": "男の人", "おんなのひと": "女の人",
    "おなじ": "同じ", "おおきい": "大きい", "ちいさい": "小さい",
    "たかい": "高い", "ひくい": "低い", "ながい": "長い", "みじかい": "短い",
    "あたらしい": "新しい", "ふるい": "古い", "おもしろい": "面白い",
    "いそがしい": "忙しい", "やさしい": "易しい", "すずしい": "涼しい",
    "うれしい": "嬉しい", "かなしい": "悲しい", "おいしい": "美味しい",
}

_KANJI_RX = re.compile(r'[\u4e00-\u9fff]')
_KANA_RX  = re.compile(r'[\u3040-\u309f\u30a0-\u30ffー]+')

_ortho_cache = None


def _ortho_data():
    """(curated, safe_kana, allowed_kanji) -- built once from the lexicon."""
    global _ortho_cache
    if _ortho_cache is not None:
        return _ortho_cache

    lex = {}
    p = os.path.join(DATA, "lexicon.json")
    if os.path.exists(p):
        try:
            lex = json.load(open(p, encoding="utf-8"))
        except Exception:
            lex = {}

    # Only demand kanji the N5 set actually permits; insisting on 欲/靴/難/暖 would
    # be the same mistake as the old JLPT vocab ceiling that we removed.
    allowed = set(lex.get("n5_kanji") or [])
    for v in lex.get("n5_vocab", []):
        allowed |= set(_KANJI_RX.findall(v.get("text") or ""))
    curated = {k: v for k, v in CURATED_KANA.items()
               if all(ch in allowed for ch in _KANJI_RX.findall(v))} if allowed else CURATED_KANA

    # Words legitimately written in kana. Must EXCLUDE readings of kanji-bearing
    # vocabulary: はこ is the reading of 箱, so counting it as a known word would
    # excuse the exact case this check exists to find.
    read_of_kanji = {v["reading"] for v in lex.get("n5_vocab", [])
                     if v.get("reading") and _KANJI_RX.search(v.get("text") or "")}
    safe = {v["text"] for v in lex.get("n5_vocab", [])
            if v.get("text") and not _KANJI_RX.search(v["text"])}
    safe |= {k for k in (lex.get("jlpt_all") or {})
             if not _KANJI_RX.search(k) and k not in read_of_kanji}
    safe -= set(curated)
    safe |= {"いくつ", "いくら", "こちら", "そちら", "あちら", "どちら",
             "おばあさん", "おじいさん", "おにいさん", "おねえさん",
             "ばあさん", "じいさん", "にいさん", "ねえさん", "ください",
             "あります", "います", "ません", "でした", "ました", "ありがとう"}

    _ortho_cache = (curated, safe, allowed)
    return _ortho_cache


def kana_violations(sentence, tok=None):
    """[(kana, kanji, start, end)] -- N5 words written in kana that have a kanji form.

    A match is rejected when it sits inside a longer kana word (くつ in いくつ,
    あさ in おばあさん) or when it crosses a token boundary (は|これ read as はこ).
    """
    if tok is None:
        tok = _get_tokenizer()
    curated, safe, _allowed = _ortho_data()
    runs = [(m.start(), m.end()) for m in _KANA_RX.finditer(sentence)]

    def inside_real_kana_word(pos, length):
        for a, b in runs:
            if a <= pos and pos + length <= b:
                for i in range(a, pos + 1):
                    for j in range(pos + length, b + 1):
                        if j - i > length and sentence[i:j] in safe:
                            return True
                return False
        return False

    # token boundaries, for the cross-boundary guard
    bounds, pos = {0}, 0
    for t in tok.tokenize(sentence):
        pos += len(t.surface)
        bounds.add(pos)

    out = []
    for kana, kanji in curated.items():
        if kanji in sentence:
            continue
        start = 0
        while True:
            p = sentence.find(kana, start)
            if p < 0:
                break
            e = p + len(kana)
            if not inside_real_kana_word(p, len(kana)) and p in bounds and e in bounds:
                out.append((kana, kanji, p, e))
                break
            start = p + 1
    return out


_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from janome.tokenizer import Tokenizer
        _TOKENIZER = Tokenizer()
    return _TOKENIZER


# ---- morphological (POS) sentence checks ------------------------------------
# Deterministic checks a string regex cannot express. Measured against our defect
# classes, POS + subcategory separates cases that are IDENTICAL as strings:
#
#   「今日は木曜日でした。」  木曜日[名詞/副詞可能] でし      -> noun predicate, OK
#   「村は静かでした。」    静か[名詞/形容動詞語幹] でし    -> na-adj, NOT Noun+でした
#
# Both match /でした/ and both pass word coverage; only the subcategory differs.
# Likewise 「買い物をします」 is 買い物[名詞/サ変接続]+を+する, where を marks a
# verbal noun rather than a direct object -- whereas 「たばこを吸います」 is
# たばこ[名詞/一般]+を+吸う, a genuine object. The subcategory carries the rule.
#
# These are GENERATION-TIME checks feeding the retry (the role check_regex plays),
# not a validation layer, and not a judge: morphology encodes no syntax, no
# semantics and no naturalness. It cannot see 「水が要りませんか」 (an invitation
# point, but a state question -- POS-identical to a real invitation) or
# 「箸を二つください」 (counted with 膳, not つ -- 箸 and りんご tokenise the same).
def _tokens(sentence):
    out = []
    for t in _get_tokenizer().tokenize(sentence):
        out.append((t.surface, t.part_of_speech.split(",")))
    return out


def predicate_is_noun(sentence):
    """The token in front of a copula must be a NOUN, not a na-adjective stem.

    Note it deliberately looks at the token BEFORE the copula, not at whether the
    sentence is well-formed: 「静かでした」 is perfectly good Japanese, it is simply
    not an example of Noun + でした.
    """
    tk = _tokens(sentence)
    for i, (surf, pos) in enumerate(tk):
        if surf in ("でし", "です", "だ") and i > 0:
            ppos = tk[i - 1][1]
            if ppos[0] != "名詞":
                return False
            if len(ppos) > 1 and ppos[1] == "形容動詞語幹":
                return False
    return True


def no_suru_setphrase(sentence):
    """Reject 名詞/サ変接続 + を + する: the を marks a verbal noun, not an object.

    This generalises the hand-written word list it replaces (買い物/洗濯/掃除/...)
    and is strictly better: it does NOT exclude 電話, so 「友だちに電話を掛けます」
    stays available while only 「電話をします」 is rejected. The old list barred the
    word outright and lost the good sentence with it.
    """
    tk = _tokens(sentence)
    for i in range(len(tk) - 2):
        a, b, c = tk[i][1], tk[i + 1][0], tk[i + 2][0]
        if (a[0] == "名詞" and len(a) > 1 and a[1] == "サ変接続"
                and b == "を" and c in ("し", "する", "す")):
            return False
    return True


POS_CHECKS = {
    "predicate_is_noun": predicate_is_noun,
    "no_suru_setphrase": no_suru_setphrase,
}


def pos_ok(sentence, names):
    """True if every named POS check accepts `sentence`."""
    for n in names or ():
        fn = POS_CHECKS.get(n)
        if fn is not None and not fn(sentence):
            return False
    return True


class Ledger:
    """Persistent usage counts, tracked GLOBALLY and per grammar point.

    Both are needed and they fix different failures:
      global    -- stops 45 noun-needing points from all opening with あさって.
                   Without this every point independently starts at the same
                   alphabetical words, so the corpus as a whole uses あさって
                   45 times. Per-job isolation caused our duplicate problem;
                   per-point isolation causes vocabulary clustering. Same bug.
      per-point -- stops one point from monopolising a word it draws well with.

    Stored as {"<gp>": {"<word>": count}} plus "__global__". Loaded and saved by
    the driver so consecutive runs keep sweeping the pool.
    """

    GLOBAL = "__global__"

    def __init__(self, path=None):
        self.path = path or os.path.join(RUNS, "ledger.json")
        self.data = {}
        if os.path.exists(self.path):
            try:
                self.data = json.load(open(self.path, encoding="utf-8"))
            except Exception:
                self.data = {}

    def count(self, gp, word):
        return self.data.get(gp, {}).get(word, 0)

    def global_count(self, word):
        return self.count(self.GLOBAL, word)

    def bump(self, gp, word):
        for k in (gp, self.GLOBAL):
            self.data.setdefault(k, {})
            self.data[k][word] = self.data[k].get(word, 0) + 1

    def draw(self, gp, pool, n, exclude=()):
        """Draw the n least-used words: globally first, then for this point.

        Ties break on the pool's natural order, so the rotation is deterministic
        and reproducible across runs.
        """
        ex = set(exclude)
        cand = [v for v in pool if v["text"] not in ex]
        # Shuffle before a STABLE sort so ties break randomly. Once every word has
        # been used the counts all tie, and sorting on (count, text) would return
        # the same alphabetical head on every run -- which is what made a single
        # good sentence (「雨が降るとき、傘をさします。」) come back nine times.
        random.shuffle(cand)
        cand.sort(key=lambda v: (self.global_count(v["text"]),
                                 self.count(gp, v["text"])))
        return cand[:n]

    def coverage(self, pool):
        """How much of a pool has been used at least once, globally."""
        used = {w for w, c in self.data.get(self.GLOBAL, {}).items() if c}
        return sum(1 for v in pool if v["text"] in used), len(pool)

    def save(self):
        json.dump(self.data, open(self.path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


# ---- prompts ----------------------------------------------------------------
# Standing rules live in `system` (weighted more heavily, survives long context).
# Per-call specifics live in `user`.

SYSTEM = """あなたは日本語教材の例文作成者です。N5（初級）レベルの短い例文だけを書きます。

必ず守ること:
- 例文は1文だけ。20文字以内。
- です・ます体（敬体）で書く。文末は です／ます／でした／ました／ません のいずれか。
- 指定された「必須語」を必ず1つ以上、そのままの形で入れる。
- 彼／彼女／あの人 を使わない。{demonstrative_rule}
- N5レベルのやさしい語彙と漢字だけを使う。
- 出力はJSONのみ。説明・思考・マークダウンは書かない。

形式: {"sentences":["例文"]}"""

# The demonstrative prohibition is omitted when the grammar point IS that form
# ("この / その / あの + Noun"), which otherwise asks the model to both use and
# avoid the same word.
DEM_RULE_OFF = ("\n"
    "- 文頭に この／その／あの／これ／それ／あれ を置かない。")

DEM_POINTS = {"kore-sore-are-demonstratives", "kono-sono-ano-dono-attributive"}


def system_for(gp):
    """System prompt for a grammar point, with the demonstrative rule dropped
    when the point exists to teach demonstratives.

    Uses replace(), not format() -- the prompt contains a literal
    {"sentences":[...]} example, and format() reads those braces as fields.
    """
    if gp in DEM_POINTS:
        return SYSTEM.replace("{demonstrative_rule}", "")
    return SYSTEM.replace("{demonstrative_rule}", DEM_RULE_OFF)


def build_user(t, required, n, avoid=None, miss=None):
    """Per-call user message.

    `required`  -- the mandatory word(s) drawn from the ledger
    `n`         -- how many sentences to produce. THIS MUST BE STATED: models do
                   not infer a count from the size of the required list, they
                   emit one sentence and stop.
    `avoid`     -- sentences already produced (dedup pressure)
    `miss`      -- the words that were not used, stated as a failure
    """
    req = "、".join(v["text"] for v in required)
    gloss = "、".join(f"{v['text']}（{v['meaning']}）" for v in required)

    lines = [
        f"文法: {t['pattern']}（{t['meaning_en']}）",
        t.get("register_instruction", ""),
        t.get("frame", ""),          # structural requirement, when a word pool cannot express it
        f"必須語: {gloss}",
        f"例文を{n}個作ってください。必須語を1つずつ、それぞれ別の文で使ってください。",
    ]

    if miss:
        lines.append(
            "前回は次の必須語が使われませんでした: " + "、".join(miss)
            + "。今回は必ずすべて使ってください。"
        )
    if avoid:
        lines.append(
            "すでに次の例文を作りました。同じ文、またはほぼ同じ文は絶対に作らないでください:\n"
            + "\n".join(f"  {s}" for s in avoid[-15:])
        )
    lines.append('出力: {"sentences":["..."]}')
    return "\n".join(x for x in lines if x)


def required_present(sentence, required, tok=None):
    """Did the sentence actually use one of the required words?

    Matches on surface OR reading OR lemma, since the generator may write the
    word in kana (e.g. 帽子 -> ぼうし) and still be using it.
    """
    for v in required:
        for form in (v["text"], v.get("reading")):
            if form and form in sentence:
                return True
        # kana-only spelling of a kanji word: compare against tokenised forms
        if tok is not None:
            forms = {t.surface for t in tok.tokenize(sentence)}
            forms |= {t.base_form for t in tok.tokenize(sentence)}
            if v["text"] in forms or (v.get("reading") and v["reading"] in forms):
                return True
    return False
