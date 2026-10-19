"""Одноразово: Milestones и Issues плана на GitHub (mamazanovfarhod016/mini-ai). Токен берётся из ~/.git-credentials."""
import json
import re
import urllib.request
from pathlib import Path

REPO = "mamazanovfarhod016/mini-ai"
tok = re.search(r"https://[^:]+:([^@]+)@github.com", Path.home().joinpath(".git-credentials").read_text()).group(1)


def api(method, path, body=None):
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}{path}", method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"token {tok}", "Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


have_ms = {m["title"]: m["number"] for m in api("GET", "/milestones?state=all&per_page=50")}
have_is = {i["title"] for i in api("GET", "/issues?state=all&per_page=100")}
MS = [("Месяц 1 — 30 млн", "2026-11-18T23:00:00Z", "Рост до ~31 млн, чтение, логика, учебники 1–6 кл., харнесс v1, приложение v1.0"),
      ("Месяц 2 — 100 млн", "2026-12-18T23:00:00Z", "Рост до ~99 млн, код на ~14 языках, CS, защита, диалоги, харнесс v2, R1 на математике"),
      ("Месяц 3 — 200 млн", "2027-01-18T23:00:00Z", "Рост до ~197 млн, окно 16k + память 100k+, R1 на коде, финальный бенчмарк")]
ISSUES = {
    0: [("Неделя 1: учебники 1–3 кл., fp16 + 2×T4, токенизатор, корпус «чтение», экзамены мышления",
         "- [ ] учебники 1–3 кл. по четвертям (90%), итог класса 95% → бенчмарк → удалить материалы\n- [ ] fp16 + 2×T4, сверка с эталоном\n- [ ] быстрый токенизатор, сверка байт в байт\n- [ ] корпус чтения 0,6 млрд токенов (fineweb-2 uz/ru, uz-books, cosmopedia, TinyStories), только открытые лицензии\n- [ ] экзамены по типам мышления + точка отсчёта"),
        ("Неделя 2: рост 6 → 31 млн, обучение «чтение», окно 2k", "- [ ] grow_params до d384×14, проверка ответов\n- [ ] чтение 0,6 млрд токенов\n- [ ] бенчмарк"),
        ("Неделя 3: учебники 4–6 кл., харнесс v1, основы Python", "- [ ] учебники 4–6 кл. + GSM8K\n- [ ] харнесс v1: calc, python (песочница), memory_search/save, read_file\n- [ ] MBPP (HumanEval только как экзамен)"),
        ("Неделя 4: самопроверка, многошаговые задачи, приложение v1.0", "- [ ] ошибка → исправление, многошаговые задачи с харнессом\n- [ ] полный экзамен + бенчмарк\n- [ ] приложение v1.0, если не хуже")],
    1: [("Рост 31 → 99 млн, словарь 16k с кодом", ""), ("Код на ~14 языках (The Stack, MIT/Apache/BSD) + CS", ""),
        ("Кибербезопасность: только защита (OWASP, CWE)", ""), ("Диалоги и понимание человека", ""),
        ("Окно 4–8k + харнесс v2", ""), ("R1 «думает сама» на математике + бенчмарк", "")],
    2: [("Рост 99 → 197 млн (деньги — только с согласия Фарходa)", ""), ("Окно 16k + память харнесса 100k+", ""),
        ("R1 на коде и исправлении уязвимостей", ""), ("Финальный экзамен и бенчмарк, отчёт для портфолио", "")],
}
for k, (title, due, desc) in enumerate(MS):
    num = have_ms.get(title) or api("POST", "/milestones", {"title": title, "due_on": due, "description": desc})["number"]
    for t, body in ISSUES[k]:
        if t not in have_is:
            api("POST", "/issues", {"title": t, "body": (body or "Подробно: docs/PLAN_3_MONTHS.md") + "\n\nПлан: ROADMAP.md",
                                    "milestone": num, "labels": ["plan"]})
print("ok", len(MS), "milestones,", sum(len(v) for v in ISSUES.values()), "issues")
