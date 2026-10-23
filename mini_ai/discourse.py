"""discourse.py — РАССУЖДЕНИЕ ПО АБЗАЦУ (discourse chain), без градиента.

Разбор промахов multi-hop (judge B, 120 вопросов, 89 промахов): в 49 случаях предложение, совпавшее с вопросом («мост»),
и предложение с ответом стоят РЯДОМ и связаны отсылкой: «…a golden-yellow liquid would be left behind. PLASMA is this fluid
part of the blood», «…cause excessive growth of algae. This is called an ALGAL BLOOM», «Bones are far from static. Instead,
they are DYNAMIC», «They have perplexing patterns… ← The TRANSITION metals are an interesting group». Отрывок (_span) смотрит
только внутрь моста и берёт мусорный кусок («behind», «field surrounding») или молчит.

Здесь — явная двухшаговая цепочка по дискурсу абзаца:
  шаг 1  мост: предложение абзаца, покрывающее больше всего веса слов вопроса;
  шаг 2  переход по связи дискурса к соседу (±1, ±2 внутри абзаца): анафора в начале соседа («This/These/It/They/Instead/
         Such/Thus/The process/One such…») → сильная связь; анафора в начале МОСТА → сосед слева — его антецедент;
         общий редкий термин → средняя; просто соседство → слабая;
  шаг 3  ответ = НОВЫЙ термин соседа (нет в вопросе): предпочтение — подлежащее определения («PLASMA is …», «MERCURY is very
         close»), объект после called/known as/due to/because of, фраза с самым редким словом; глагольные хвосты — штраф.
Мысли пишутся как текст по шагам — это и есть объяснение ответа. Уверенность = покрытие моста × сила связи × качество образца,
так что прямой отрывок с хорошим куском всё равно выигрывает, а цепочка закрывает молчание и мусорные куски.
"""
import re, collections
from mini_ai.memory_v2 import toks, content_words, STOP

ANAPH = re.compile(r"^(?:this|these|that|those|it|they|its|their|such|thus|instead|here|there|one such|such a|such an|the process|the term|the name|the result|this process|this is|these are|in this|in these|as a result|for this reason|because of this|therefore|so)\b", re.I)
DEF_AFTER = re.compile(r"\b(?:is|are|was|were)\s+(?:called|known as|termed|named|referred to as)\s+(?:a |an |the )?", re.I)
CAUSE_AFTER = re.compile(r"\b(?:due to|because of|caused by|as a result of|made (?:up )?of|composed of|consists? of)\s+(?:a |an |the )?", re.I)
COPULA = {"is", "are", "was", "were"}
JUNK = set("today now then also usually often sometimes generally typically however thus therefore example instance figure below above here there several many some few most all each every both another other such one two three first second last next various numerous large small very more less".split())


class Discourse:
    def __init__(self, mind):
        self.mind = mind; self.kb = mind.kb

    # ---------- служебное ----------
    def _para(self, si):
        sp = getattr(self.kb, "sent_para", None)
        return sp[si] if sp and si < len(sp) else None

    def _words(self, s): return {w for w in content_words(toks(s)) if w not in STOP}

    def _forms_of(self, ws):
        out = set()
        for w in ws: out |= self.mind._forms(w)
        return out

    # ---------- шаг 1: мосты ----------
    def bridges(self, qw, k=4):
        """предложения, покрывающие больше всего веса слов вопроса → [(si, cov)]"""
        kb = self.kb; idf = {w: self.mind._widf(w) for w in qw}; tot = sum(idf.values()) or 1.0
        cnt = collections.Counter(); forms = {w: self.mind._forms(w) for w in qw}
        for w in qw:
            alts = {w} | ({x for x in kb.sent_by_word if len(x) > 3 and (self.mind._forms(x) & forms[w])} if len(w) > 3 else set())
            seen = set()
            for x in alts:
                for si in kb.sent_by_word.get(x, []):
                    if si not in seen: cnt[si] += idf[w] * (1.0 if x == w else 0.9); seen.add(si)
        return [(si, sc / tot) for si, sc in cnt.most_common(k)]

    # ---------- шаг 2: связь дискурса ----------
    def link(self, bi, nj):
        """сила связи мост → сосед (0..1) и её словесное имя"""
        if self._para(bi) is not None and self._para(bi) != self._para(nj): return 0.0, None
        if abs(bi - nj) > 2: return 0.0, None
        sb, sn = self.kb.sents[bi][0], self.kb.sents[nj][0]
        wb, wn = self._words(sb), self._words(sn)
        shared = {w for w in (self._forms_of(wb) & self._forms_of(wn)) if len(w) > 3}
        if nj == bi + 1 and ANAPH.match(sn): return 1.0, "the next sentence refers back to it"
        if nj == bi - 1 and ANAPH.match(sb): return 0.95, "it refers back to the previous sentence"
        if nj == bi + 1 and re.match(r"^(?:in other words|that is|i\.e\.|for example|namely)\b", sn, re.I): return 0.9, "the next sentence restates it"
        if abs(bi - nj) == 1 and shared: return 0.65, "the neighbouring sentence shares the term «%s»" % sorted(shared, key=len)[-1]
        if abs(bi - nj) == 1: return 0.4, "it is the neighbouring sentence"
        if shared and (ANAPH.match(sn) or ANAPH.match(sb)): return 0.5, "two sentences on, linked by «%s»" % sorted(shared, key=len)[-1]
        return 0.0, None

    # ---------- шаг 3: новый термин соседа ----------
    def candidates(self, sent, exclude_forms, qmeta):
        """фразы-кандидаты из предложения: (score, phrase, why). Исключаются слова вопроса; фразы — пробеги не-стоп слов,
        соединённые через «of»; бонусы: подлежащее-определение, после called/known as/due to; штраф глагольным хвостам."""
        body = re.sub(r"\s*\(([^)]*)\)", r", \1, ", sent)
        tl = toks(body); out = []; i = 0
        low = body.lower()
        while i < len(tl):
            if tl[i] in STOP or tl[i] in JUNK or tl[i] in qmeta: i += 1; continue
            run = []; j = i
            while j < len(tl) and (tl[j] not in STOP and tl[j] not in JUNK and tl[j] not in qmeta or (tl[j] == "of" and run and j + 1 < len(tl) and tl[j + 1] not in STOP)):
                run.append(tl[j]); j += 1
            if run and run[-1] == "of": run.pop(); j -= 1
            words = [w for w in run if w != "of"]
            if not words or not any(w.isalpha() for w in words): i = j; continue
            new = [w for w in words if not (self.mind._forms(w) & exclude_forms)]
            if not new or len(words) > 6: i = j; continue
            # фраза берётся целиком, но её «новизна» = доля новых слов; фразы без новых слов — не ответ
            phrase = " ".join(run)
            rare = max(self.kb.idf(w) for w in new); sc = rare * (len(new) / len(words)) ** 0.5
            why = "rare term"
            prev = tl[i - 1] if i > 0 else None; nxt = tl[j] if j < len(tl) else None
            start = i == 0 or (i == 1 and tl[0] in ("the", "a", "an"))
            if start and nxt in COPULA: sc += 2.2; why = "subject of a definition"
            pre = " ".join(tl[max(0, i - 4):i])
            if DEF_AFTER.search(pre + " "): sc += 2.6; why = "named after «called/known as»"
            elif CAUSE_AFTER.search(pre + " "): sc += 2.0; why = "named after «due to/because of/made of»"
            elif prev in COPULA and i >= 2 and (tl[i - 2] in ("instead", "they", "it", "these", "this") or (i >= 3 and tl[i - 3] in ("instead",))): sc += 1.6; why = "predicate of the back-reference"
            elif prev in ("a", "an") and i >= 2 and tl[i - 2] in COPULA: sc += 1.2; why = "class after «is a»"
            verbish = all(w.endswith(("ed", "ing", "ly")) or w in ("made", "used", "found", "done", "seen", "known", "left", "take", "takes", "make", "makes") for w in new)
            if verbish: sc *= 0.3
            if all(w.isdigit() for w in words): sc *= 0.2
            out.append((round(sc, 3), phrase, why)); i = j
        return sorted(out, key=lambda c: -c[0])

    # ---------- вся цепочка ----------
    def reason(self, q, fr, max_bridges=4):
        """→ dict(answer, confidence, thoughts, facts, bridge, neighbor) | None"""
        kb = self.kb
        if not kb.sents: return None
        known = set(fr.get("X") or []) | set(fr.get("Y") or [])
        qw = {w for w in known if w not in self.mind.META and w not in STOP} | {w for w in content_words(toks((fr.get("rel") or "").replace("_", " "))) if w not in self.mind.META and w not in ("prop", "called", "be", "do", "have")}
        if not qw: return None
        qforms = self._forms_of(self._words(q)) | self._words(q)
        qmeta = {w for w in self._words(q) if w in self.mind.META}
        best = None
        for bi, cov in self.bridges(qw, k=max_bridges):
            if cov < 0.3: continue
            sb = kb.sents[bi][0]
            bforms = self._forms_of(self._words(sb))
            for nj in (bi + 1, bi - 1, bi + 2, bi - 2):
                if nj < 0 or nj >= len(kb.sents): continue
                lk, lname = self.link(bi, nj)
                if lk <= 0: continue
                sn = kb.sents[nj][0]
                cands = self.candidates(sn, qforms, qmeta)
                if not cands: continue
                sc, phrase, why = cands[0]
                quality = 1.0 if sc >= 2.0 + 3.0 else (0.75 if sc >= 4.0 else 0.5)
                conf = min(0.93, min(1.0, cov) * lk * (0.5 + 0.45 * quality))
                if lk >= 0.9 and why in ("named after «called/known as»", "subject of a definition"): conf = max(conf, min(0.93, 0.6 + 0.3 * min(1.0, cov)))
                cand = dict(answer=phrase, confidence=round(conf, 2), bridge=sb, neighbor=sn, link=lname, why=why, cov=round(cov, 2), sc=sc)
                if not best or cand["confidence"] > best["confidence"] or (cand["confidence"] == best["confidence"] and sc > best["sc"]): best = cand
        if not best: return None
        ans = re.sub(r"^(?:the|a|an|its|their|also|about|only|all|most|many|some|very)\s+", "", best["answer"]).strip()
        if len(ans) < 2: return None
        best["answer"] = ans
        best["thoughts"] = [f"Step 1: the question matches «{best['bridge'].rstrip('.')}».",
                            f"Step 2: {best['link']}: «{best['neighbor'].rstrip('.')}».",
                            f"Step 3: the new term there is «{ans}» ({best['why']}).", f"So: {ans}."]
        best["facts"] = [f"{ans} —discourse→ {best['neighbor'][:120]}", f"bridge —{best['link']}→ {best['bridge'][:120]}"]
        return best
