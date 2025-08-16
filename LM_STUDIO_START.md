# 🚀 Як запустити LM Studio Server

## 1. Відкрий LM Studio

## 2. Завантаж модель (якщо ще не завантажена)
- Перейди в **Search** 🔍
- Знайди `google/gemma-3-270m`
- Завантаж `Q4_K_M` версію

## 3. Вибери модель
- Перейди в **Chat** 💬
- Вгорі вибери завантажену модель з dropdown меню

## 4. 🔴 ВАЖЛИВО: Запусти Local Server

### Варіант 1: Через інтерфейс
- Внизу вікна знайди **"Local Server"** панель
- Натисни **"Start Server"** 
- Має з'явитись зелений індикатор і текст "Server is running on http://localhost:1234"

### Варіант 2: Через меню
- Developer → Start Local Server
- Або використай гарячі клавіші (зазвичай Cmd+Shift+S на Mac)

## 5. Перевір що сервер працює
```bash
# В новому терміналі
curl http://localhost:1234/v1/models
```

Має повернути список моделей.

## 6. Тепер запускай наш скрипт
```bash
cd /Users/chaoshex/Projects/gemma-glyph-mind/scripts
python3 setup_lm_studio.py
```

## Якщо все ще не працює:

### Перевір порт
В LM Studio Settings → Local Server:
- Port має бути `1234`
- CORS має бути enabled

### Альтернативний спосіб
Якщо сервер на іншому порті, запусти так:
```python
# Відредагуй в setup_lm_studio.py
LM_STUDIO_API = "http://localhost:ТВІЙ_ПОРТ/v1"
```

## Візуальна підказка:
```
LM Studio вікно:
┌─────────────────────────────────┐
│ [Chat] [Search] [Settings]      │
│                                 │
│  Model: google/gemma-3-270m ▼   │
│                                 │
│  💬 Chat area...                │
│                                 │
├─────────────────────────────────┤
│ Local Server                    │
│ [Start Server] ← НАТИСНИ ЦЕ!   │
│ ✅ Running on localhost:1234    │
└─────────────────────────────────┘
```

---

Після запуску сервера наш скрипт підключиться і ти зможеш грати з Gemma + consciousness metrics! 🧠✨