# Бенчмарки Mini AI

Официальные тестовые части; модель их в учёбе не видит. Случайное угадывание: 25% (4 варианта), GSM8K около 0%. Метод для вариантов — как в EleutherAI lm-evaluation-harness (вероятность ответа).

| Когда | Этап | Параметров | ARC-Easy | ARC-Challenge | MMLU elementary_mathematics | MMLU high_school_mathematics | MMLU high_school_physics | MMLU high_school_computer_science | MMLU college_computer_science | MMLU computer_security | GSM8K en | GSM8K ru |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-19 10:27 | точка отсчёта (v0.5, до учёбы по книгам) | 3.0 млн | 22.0% | 26.4% | 25.7% | 20.4% | 19.9% | 19.0% | 23.0% | 17.0% | 1.0% | 1.0% |
| 2026-10-20 01:33 | после 1 класса | 6.6 млн | 24.4% | 26.2% | 25.4% | 23.7% | 19.2% | 19.0% | 26.0% | 15.0% | 1.5% | 1.0% |
