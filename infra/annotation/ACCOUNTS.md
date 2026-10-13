# Аккаунты AGY

Проверено через API Google: 2026-10-13T09:03:15+00:00.

Тариф берётся из `paidTier`. Лимит ниже — наша настройка параллельности, не квота Google. Остатки квоты смотрим через `check_usage.py`; они меняются во время работы.

| Слот | Почта | Подписка | Подключение | Refresh token сохранён | Лимит запросов |
|---|---|---|---|---|---:|
| agy-01 | elejatoqokim01@gmail.com | Google AI Pro | Первоначальный способ не подтверждён: OAuth или импорт refresh token | Да | 5 |
| agy-02 | asuxoyikokub51@gmail.com | Google AI Pro | Первоначальный способ не подтверждён: OAuth или импорт refresh token | Да | 5 |
| agy-03 | anhlanguyen123abcxyz@gmail.com | Google AI Ultra | Первоначальный способ не подтверждён: OAuth или импорт refresh token | Да | 20 |
| agy-04 | zepunowu59@gmail.com | Antigravity Starter Quota | Первоначальный способ не подтверждён: OAuth или импорт refresh token | Да | 5 |
| agy-05 | art0644857923@gmail.com | Google AI Ultra | OAuth в браузере, добавлен 13.10.2026 | Да | 20 |
| agy-06 | art0644854407@gmail.com | Google AI Ultra | OAuth в браузере, добавлен 13.10.2026 | Да | 20 |
| agy-07 | mohammed738255602@gmail.com | Google AI Ultra | Первоначально скопирована действующая авторизация с хоста; см. README | Да | 20 |

Все семь аккаунтов используют `auth_method=consumer`. OAuth также сохраняет refresh token; поэтому по текущему файлу авторизации нельзя отличить вход через браузер от импорта предоставленного токена. Неустановленное происхождение отмечено явно.

## Расположение и проверка

- Сервер: `dev-1`, hostname `exp-srv-rtx6000x4-02`.
- Сервис каждого слота: `agy-01` … `agy-07`; прокси: соответствующий `agy-XX-proxy`.
- Авторизация изолирована в home volume каждого слота. Путь внутри контейнера: `/home/worker/.gemini/antigravity-cli/antigravity-oauth-token`.
- Токены, OAuth-коды, пароли и VLESS-ссылки в эту таблицу и Git не помещать.
- 04 ранее считался Pro по предположению, но API при этой проверке подтвердил **Antigravity Starter Quota**.
- У 05 и 06 разные Google subject/email; близкие значения usage не доказывают независимость квот.

```bash
python3 infra/annotation/check_usage.py --output /tmp/agy-usage-latest.json
```

При добавлении/замене аккаунта обновить почту, подтверждённый тариф, происхождение авторизации и дату проверки. При временной квоте аккаунт остаётся зарегистрированным; текущую активность проверять отдельно.
