from google import genai
from google.genai import types
from google.api_core import exceptions as core_exceptions
import json
import os
import time
import random

DATASET_JSON_FILE = "training_dataset.json"
BASE_MODEL_FOR_TUNING = 'models/gemini-1.5-flash-001-tuning'
TUNED_MODEL_DISPLAY_NAME = "youtube-comment-generator-v25"
EPOCH_COUNT = 6
BATCH_SIZE = 4
LEARNING_RATE = 0.0003
POLLING_INTERVAL_SECONDS = 60

print("--- Запуск скрипта Fine-Tuning Gemini ---")

print(f"1. Создание клиента Gemini (ожидается ключ из GOOGLE_API_KEY)...")
try:
    client = genai.Client()
    client.models.list()
    print("   Клиент успешно создан и аутентифицирован.")
except Exception as e:
    print(f"   Ошибка при создании клиента или проверке аутентификации: {e}")
    print("   Убедитесь, что переменная окружения GOOGLE_API_KEY установлена правильно.")
    exit()

print(f"\n2. Загрузка датасета из '{DATASET_JSON_FILE}'...")
if not os.path.exists(DATASET_JSON_FILE):
    print(f"   ОШИБКА: Файл датасета '{DATASET_JSON_FILE}' не найден!")
    exit()

raw_dataset = []
try:
    with open(DATASET_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
             if all(isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str) and isinstance(pair[1], str) for pair in data):
                 raw_dataset = data
             else:
                 print("   ОШИБКА: Неверная структура данных в JSON (ожидается список списков [[вход, выход], ...]).")
                 exit()
        elif isinstance(data, dict) and "dataset" in data and isinstance(data["dataset"], list):
            if all(isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str) and isinstance(pair[1], str) for pair in data["dataset"]):
                raw_dataset = data["dataset"]
            else:
                 print("   ОШИБКА: Неверная структура данных в ключе 'dataset' (ожидается список списков [[вход, выход], ...]).")
                 exit()
        else:
            print("   ОШИБКА: Файл JSON должен быть списком списков [[вход, выход], ...] или содержать ключ 'dataset' с таким списком.")
            exit()

    print(f"   Датасет успешно загружен. Общее количество примеров: {len(raw_dataset)}")

except json.JSONDecodeError as e:
    print(f"   ОШИБКА при разборе JSON: {e}")
    exit()
except Exception as e:
    print(f"   ОШИБКА при чтении файла: {e}")
    exit()

if not raw_dataset:
    print("   ОШИБКА: Датасет пуст.")
    exit()

print("\n3. Форматирование всего датасета для обучения Google AI...")
try:
    training_dataset_formatted = types.TuningDataset(
        examples=[
            types.TuningExample(text_input=pair[0], output=pair[1])
            for pair in raw_dataset
        ]
    )
    print(f"   Обучающий датасет отформатирован. Примеров: {len(training_dataset_formatted.examples)}")
    if len(training_dataset_formatted.examples) != len(raw_dataset):
        print("   ВНИМАНИЕ: Некоторые примеры могли быть отфильтрованы при форматировании.")

except Exception as e:
    print(f"   ОШИБКА форматирования: {e}")
    exit()

if not training_dataset_formatted or not training_dataset_formatted.examples:
    print("   ОШИБКА: Отформатированный обучающий датасет пуст.")
    exit()

print("\n4. Конфигурация задачи Fine-Tuning...")
tuning_config = types.CreateTuningJobConfig(
    epoch_count=EPOCH_COUNT,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME
)
print(f"   Базовая модель: {BASE_MODEL_FOR_TUNING}")
print(f"   Параметры: Эпохи={EPOCH_COUNT}, Пакет={BATCH_SIZE}, LR={LEARNING_RATE}")
print(f"   Имя модели: {TUNED_MODEL_DISPLAY_NAME}")

print("\n5. Запуск задачи Fine-Tuning (это может занять время)...")
tuning_job = None
try:
    tuning_job = client.tunings.tune(
        base_model=BASE_MODEL_FOR_TUNING,
        training_dataset=training_dataset_formatted,
        config=tuning_config
    )
    print(f"\n   Задача Fine-Tuning запущена!")
    print(f"   Имя задачи (Tuning Job Name): {tuning_job.name}")
    print(f"   Имя готовой модели (когда завершится): Определится позже")

except core_exceptions.GoogleAPIError as e:
     print(f"\n   ОШИБКА API при запуске задачи Fine-Tuning: {e}")
     if hasattr(e, 'message'): print(f"   Сообщение: {e.message}")
     exit()
except TypeError as e:
     print(f"\n   ОШИБКА ТИПА при запуске задачи Fine-Tuning: {e}")
     exit()
except Exception as e:
    print(f"\n   НЕПРЕДВИДЕННАЯ ОШИБКА при запуске задачи Fine-Tuning: {e}")
    exit()

if not tuning_job:
    print("   Не удалось запустить задачу тюнинга.")
    exit()

print(f"\n6. Мониторинг статуса задачи '{tuning_job.name}' (проверка каждые {POLLING_INTERVAL_SECONDS} сек)...")
final_job_status = None
tuned_model_name = None
start_time = time.time()

while True:
    try:
        job_status = client.tunings.get(name=tuning_job.name)
        final_job_status = job_status

        current_state_name = "UNKNOWN"
        if job_status.state:
             current_state_name = job_status.state.name

        elapsed_time = time.time() - start_time
        print(f"   {time.strftime('%H:%M:%S')} | Статус: {current_state_name} | Прошло времени: {elapsed_time:.0f} сек")

        if current_state_name == 'JOB_STATE_SUCCEEDED':
            tuned_model_info = getattr(job_status, 'tuned_model', None)
            if tuned_model_info and hasattr(tuned_model_info, 'model'):
                 tuned_model_name = tuned_model_info.model
                 print(f"\n   ЗАДАЧА УСПЕШНО ЗАВЕРШЕНА!")
                 print(f"   Имя готовой модели: {tuned_model_name}")
            else:
                 print("\n   ЗАДАЧА ЗАВЕРШЕНА (SUCCEEDED), но не удалось получить имя дообученной модели из ответа API.")
                 print(f"   Полный ответ job_status: {job_status}")
            break

        elif current_state_name in ['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']:
            print(f"\n   ОШИБКА/ОТМЕНА: Задача завершилась со статусом {current_state_name}.")
            error_info = getattr(job_status, 'error', None)
            if error_info:
                 print(f"   Код ошибки: {getattr(error_info, 'code', 'N/A')}")
                 print(f"   Сообщение: {getattr(error_info, 'message', 'N/A')}")
            else:
                 print(f"   Дополнительная информация об ошибке недоступна. Полный ответ job_status: {job_status}")
            break

        elif current_state_name in ['JOB_STATE_UNSPECIFIED', 'JOB_STATE_PENDING', 'JOB_STATE_RUNNING', 'JOB_STATE_COMPLETED']:
             time.sleep(POLLING_INTERVAL_SECONDS)
        else:
            print(f"\n   Неизвестный статус задачи: {current_state_name}. Прерывание мониторинга.")
            print(f"   Полный ответ job_status: {job_status}")
            break

    except core_exceptions.NotFound:
        print(f"\n   ОШИБКА: Задача тюнинга с именем '{tuning_job.name}' не найдена. Возможно, она была удалена.")
        break
    except core_exceptions.GoogleAPIError as e:
        print(f"\n   ОШИБКА API во время мониторинга: {e}. Повторная попытка через {POLLING_INTERVAL_SECONDS} сек.")
        time.sleep(POLLING_INTERVAL_SECONDS)
    except TypeError as e:
        print(f"\n   ОШИБКА ТИПА во время мониторинга (возможно, проблема с аргументами вызова): {e}. Прерывание.")
        break
    except Exception as e:
        print(f"\n   НЕПРЕДВИДЕННАЯ ОШИБКА во время мониторинга: {e}. Прерывание.")
        if final_job_status:
             print(f"   Последний полученный статус: {final_job_status}")
        break

if tuned_model_name:
    print(f"\n7. Обучение завершено. Имя модели: {tuned_model_name}")
    print("   Используйте скрипт оценки для проверки качества.")
else:
    print("\n7. Обучение завершилось, но имя модели не было получено.")

print("\n--- Скрипт Fine-Tuning завершен ---")