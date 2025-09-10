import json
import os
import re
import time
import google.generativeai as genai
from google.generativeai.types import generation_types
from colorama import init, Fore, Style

# -- Конфигурация --
INPUT_FILENAME = "training_dataset.json"
OUTPUT_FILENAME = "processed_training_dataset.json"
# --- ВАЖНО: Проверь имя модели ---
# Ты указал gemini-2.0-flash-001. Такой модели пока нет в общем доступе.
# Использую последнюю известную flash-модель: gemini-1.5-flash-001.
# Если у тебя есть доступ к 2.0, ИЗМЕНИ строку ниже на "models/gemini-2.0-flash-001"
MODEL_NAME = "models/gemini-2.0-flash-001"
# ----------------------------------
DESC_LENGTH_THRESHOLD = 300
API_DELAY_SECONDS = 2  # Задержка между запросами к API
MAX_CONSECUTIVE_ERRORS = 3

# -- Инициализация Colorama для цветного вывода --
init(autoreset=True)

# -- Функция для извлечения метаданных --
def extract_metadata(text):
    """Извлекает Title, Channel, Description и Comments из строки метаданных."""
    title_match = re.search(r"Title: (.*?)\n", text)
    channel_match = re.search(r"Channel: (.*?)\n", text)
    description_match = re.search(r"Description:\n(.*?)\n--- Comments ---", text, re.DOTALL)
    comments_match = re.search(r"--- Comments ---\n(.*)", text, re.DOTALL)

    title = title_match.group(1).strip() if title_match else None
    channel = channel_match.group(1).strip() if channel_match else None
    # Если description_match пустой, вернем пустую строку вместо None
    description = description_match.group(1).strip() if description_match else ""
    # Если comments_match пустой, вернем пустую строку
    comments_section = comments_match.group(1).strip() if comments_match else ""

    # Проверяем только обязательные поля (например, Title и Channel)
    if not title or not channel:
         print(Fore.YELLOW + f"Предупреждение: Не удалось распарсить Title или Channel в метаданных:\n{text[:200]}...")
         return None, None, None, None # Сигнализируем о пропуске

    return title, channel, description, comments_section

# -- Функция для очистки комментариев --
def clean_comments(comments_text):
    """Удаляет '(ID: ...)' из текста комментариев."""
    if not comments_text:
        return ""
    cleaned_text = re.sub(r"\s*\(ID: [a-zA-Z0-9_]+\)", "", comments_text)
    return cleaned_text

# -- Функция для сокращения описания через Gemini --
def shorten_description_gemini(model_client, title, channel, original_description):
    """Отправляет запрос в Gemini API для сокращения описания."""
    prompt = f"""
Сократи следующее описание видео с YouTube, учитывая его название и название канала. Оставь только самую суть, удали:
- Призывы к действию (подписаться, поставить лайк, перейти по ссылкам).
- Рекламные интеграции или упоминания спонсоров.
- Ссылки на социальные сети, другие каналы, плейлисты (если они не являются основной темой видео, указанной в названии).
- Неинформативные или повторяющиеся теги, перечисленные в описании, особенно если они явно не связаны с названием видео или каналом.
- Приветствия, прощания, общие фразы.
- Тайм-коды (если они не критичны для понимания сути).

Сохрани ключевую информацию о содержании видео, как указано в названии и подкреплено описанием. Игнорируй теги, если они противоречат названию видео. Ответ должен быть только текстом сокращенного описания, без твоих комментариев до или после.

Название видео: {title}
Канал: {channel}

Оригинальное описание:
---
{original_description}
---

Сокращенное описание:
"""
    try:
        print(Fore.MAGENTA + f"Отправка запроса на СОКРАЩЕНИЕ (Видео: '{title}') в Gemini (модель: {MODEL_NAME})...")
        # Настройки безопасности (можно настроить или убрать)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = model_client.generate_content(
            contents=[prompt],
            safety_settings=safety_settings,
            # generation_config=generation_types.GenerationConfig(temperature=0.2) # Можно раскомментировать
        )

        # Проверка ответа (более надежная)
        if response.parts:
            summary = response.text.strip()
            if summary:
                print(Fore.GREEN + "Сокращенное описание получено от Gemini.")
                return summary, None # Успех
            else:
                # Случай, когда есть parts, но text пустой (маловероятно, но возможно)
                reason = "Ответ Gemini пуст, хотя структура ответа корректна."
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     reason = f"Заблокировано Gemini: {response.prompt_feedback.block_reason.name} ({response.prompt_feedback.block_reason_message or 'Нет сообщения'})"
                print(Fore.YELLOW + f"Gemini вернул пустой текст. {reason}")
                return None, reason
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             # Случай, когда ответ заблокирован до генерации
             reason = f"Заблокировано Gemini: {response.prompt_feedback.block_reason.name} ({response.prompt_feedback.block_reason_message or 'Нет сообщения'})"
             print(Fore.YELLOW + f"Gemini не вернул текст. {reason}")
             return None, reason
        else:
             # Другие случаи пустого ответа (нет ни parts, ни block_reason)
             reason = "Неизвестная причина пустого ответа от Gemini."
             print(Fore.YELLOW + f"Gemini не вернул текст. {reason}")
             return None, reason

    except generation_types.BlockedPromptException as bpe:
         print(Fore.RED + f"Запрос заблокирован Gemini (BlockedPromptException): {bpe}")
         return None, f"Заблокировано Gemini (Исключение): {bpe}"
    except generation_types.StopCandidateException as sce:
         print(Fore.RED + f"Генерация остановлена Gemini (StopCandidateException): {sce}")
         return None, f"Остановлено Gemini (Исключение): {sce}"
    except Exception as e:
        # Ловим другие возможные ошибки API
        print(Fore.RED + f"Ошибка при вызове Gemini API (summarize): {type(e).__name__}: {e}")
        # Попытка извлечь более детальную информацию, если она есть
        error_message = str(e)
        if hasattr(e, 'message'):
             error_message = e.message # Иногда более информативно
        return None, f"Ошибка Gemini API: {error_message}"

# -- Основной скрипт --
if __name__ == "__main__":
    print(Fore.CYAN + "--- Запуск скрипта обработки датасета ---")

    # 1. Настройка клиента Gemini
    gemini_model = None
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Переменная окружения GOOGLE_API_KEY не установлена.")

        # Конфигурируем API ключ глобально
        genai.configure(api_key=api_key)

        # Создаем экземпляр модели
        gemini_model = genai.GenerativeModel(MODEL_NAME)

        # Простая проверка доступности модели (пытаемся сгенерировать 1 токен)
        print(f"Проверка доступности модели '{MODEL_NAME}'...")
        gemini_model.generate_content("test", generation_config=generation_types.GenerationConfig(max_output_tokens=1, candidate_count=1))
        print(Fore.GREEN + f"Модель '{MODEL_NAME}' доступна и готова к работе.")

    except ValueError as ve:
        print(Fore.RED + f"Ошибка конфигурации: {ve}")
        exit(1)
    except Exception as e:
        print(Fore.RED + f"Ошибка при инициализации или проверке модели Gemini: {type(e).__name__}: {e}")
        print(Fore.YELLOW + "Убедитесь, что:")
        print(Fore.YELLOW + "  - API ключ GOOGLE_API_KEY установлен и действителен.")
        print(Fore.YELLOW + f"  - Модель '{MODEL_NAME}' существует и доступна для вашего ключа.")
        print(Fore.YELLOW + "  - У вас установлена последняя версия библиотеки: pip install -U google-generativeai")
        exit(1)

    # 2. Загрузка данных
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "dataset" not in data or not isinstance(data["dataset"], list):
                 raise ValueError("Некорректная структура JSON: отсутствует ключ 'dataset' или он не является списком.")
            original_dataset = data["dataset"]
        print(Fore.GREEN + f"Успешно загружен файл '{INPUT_FILENAME}'. Записей: {len(original_dataset)}")
    except FileNotFoundError:
        print(Fore.RED + f"Ошибка: Файл '{INPUT_FILENAME}' не найден.")
        exit(1)
    except json.JSONDecodeError as jde:
        print(Fore.RED + f"Ошибка: Не удалось декодировать JSON из файла '{INPUT_FILENAME}'. {jde}")
        exit(1)
    except ValueError as ve:
        print(Fore.RED + f"Ошибка структуры данных: {ve}")
        exit(1)
    except Exception as e:
        print(Fore.RED + f"Непредвиденная ошибка при загрузке файла: {type(e).__name__}: {e}")
        exit(1)


    # 3. Обработка данных
    processed_dataset = []
    consecutive_errors = 0
    processed_count = 0
    skipped_count = 0
    shortened_count = 0
    total_records = len(original_dataset)

    for index, item in enumerate(original_dataset):
        print(Style.BRIGHT + f"\n--- Обработка записи {index + 1}/{total_records} ---")
        if not isinstance(item, list) or len(item) != 2:
            print(Fore.YELLOW + f"Предупреждение: Некорректный формат элемента {index}. Ожидался список из двух строк. Пропуск.")
            skipped_count += 1
            continue

        metadata_string, target_comment = item
        if not isinstance(metadata_string, str) or not isinstance(target_comment, str):
             print(Fore.YELLOW + f"Предупреждение: Элемент {index} содержит нестроковые данные. Пропуск.")
             skipped_count +=1
             continue

        # Извлечение метаданных
        title, channel, description, comments_section = extract_metadata(metadata_string)

        if title is None: # Если парсинг не удался (extract_metadata вернула None)
             print(Fore.YELLOW + f"Предупреждение: Не удалось извлечь обязательные метаданные для элемента {index}. Пропуск.")
             skipped_count += 1
             continue

        # Очистка комментариев
        cleaned_comments = clean_comments(comments_section)

        # Обработка описания
        final_description = description # По умолчанию используем оригинальное (может быть "" если не нашлось)
        if len(description) >= DESC_LENGTH_THRESHOLD:
            print(f"Описание для '{title}' ({len(description)} символов) требует сокращения.")

            # Вызов Gemini
            summary, error_reason = shorten_description_gemini(gemini_model, title, channel, description)

            if summary is not None: # Проверяем что summary не None (успех)
                final_description = summary
                consecutive_errors = 0 # Сбрасываем счетчик ошибок при успехе
                shortened_count += 1
            else:
                # Ошибка или пустой ответ
                print(Fore.YELLOW + f"Не удалось сократить описание для '{title}'. Причина: {error_reason}. Будет использовано оригинальное описание.")
                consecutive_errors += 1
                # final_description остается оригинальным (description)

            # Проверка на максимальное количество последовательных ошибок
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(Fore.RED + f"Критическая ошибка: Достигнуто максимальное количество последовательных ошибок API ({MAX_CONSECUTIVE_ERRORS}). Прерывание обработки.")
                break # Выход из цикла for

            # Задержка перед следующим запросом (даже если была ошибка)
            print(f"Пауза {API_DELAY_SECONDS} сек...")
            time.sleep(API_DELAY_SECONDS)

        else:
            print(f"Описание для '{title}' ({len(description)} символов) короткое, сокращение не требуется.")
            # final_description уже равен description

        # Сборка обновленной строки метаданных
        updated_metadata_string = (
            f"Title: {title}\n"
            f"Channel: {channel}\n"
            f"Description:\n{final_description}\n"  # Используем final_description
            f"--- Comments ---\n{cleaned_comments}" # Используем очищенные комменты
        )

        processed_dataset.append([updated_metadata_string, target_comment])
        processed_count += 1

    # 4. Сохранение данных
    if consecutive_errors < MAX_CONSECUTIVE_ERRORS:
        try:
            output_data = {"dataset": processed_dataset}
            with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False) # indent=2 для экономии места
            print(Fore.GREEN + Style.BRIGHT + f"\n--- Обработка успешно завершена ---")
            print(f"Всего записей в исходном файле: {total_records}")
            print(f"Обработано записей: {processed_count}")
            print(f"Сокращено описаний: {shortened_count}")
            print(f"Пропущено записей (ошибки парсинга/формата): {skipped_count}")
            print(Fore.GREEN + f"Результат сохранен в файл '{OUTPUT_FILENAME}'.")
        except Exception as e:
            print(Fore.RED + f"Критическая ошибка при сохранении результата в файл '{OUTPUT_FILENAME}': {type(e).__name__}: {e}")
    else:
         # Этот блок выполняется, если вышли из цикла из-за ошибок API
         print(Fore.RED + Style.BRIGHT + f"\n--- Обработка прервана из-за ошибок API ---")
         print(f"Всего записей в исходном файле: {total_records}")
         print(f"Обработано записей до прерывания: {processed_count}")
         print(f"Сокращено описаний: {shortened_count}")
         print(f"Пропущено записей: {skipped_count}")
         print(Fore.YELLOW + f"Результат НЕ был сохранен в файл '{OUTPUT_FILENAME}'. Исправьте проблему с API и запустите скрипт заново.")

# -- Деинициализация Colorama (обычно не нужна в скриптах) --
# init(autoreset=True) # Достаточно одного раза в начале