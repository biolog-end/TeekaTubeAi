from google import genai
from google.api_core import exceptions as core_exceptions
import json
import os
import random

FINE_TUNED_MODEL_NAME = "tunedModels/youtubecommentgeneratorv1-9fcq06wpeo1j" # Замените на имя вашей модели, например tunedModels/my-model-1234
DATASET_JSON_FILE = "training_dataset.json" # Путь к оригинальному датасету для взятия примеров
NUM_EXAMPLES_TO_TEST = 5
RANDOM_SEED = 42 # Фиксированный сид для воспроизводимости

print("--- Запуск скрипта Оценки Модели ---")


print(f"1. Создание клиента Gemini...")
try:
    client = genai.Client()
    client.models.list()
    print("   Клиент успешно создан и аутентифицирован.")
except Exception as e:
    print(f"   Ошибка при создании клиента: {e}")
    exit()

print(f"\n2. Загрузка датасета из '{DATASET_JSON_FILE}' для тестов...")
if not os.path.exists(DATASET_JSON_FILE):
    print(f"   ОШИБКА: Файл датасета '{DATASET_JSON_FILE}' не найден!")
    exit()

raw_dataset = []
try:
    with open(DATASET_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            raw_dataset = data
        elif isinstance(data, dict) and "dataset" in data and isinstance(data["dataset"], list):
            raw_dataset = data["dataset"]
        else:
             print("   ОШИБКА: Неверный формат JSON файла.")
             exit()
    print(f"   Датасет загружен. Всего примеров: {len(raw_dataset)}")
except Exception as e:
    print(f"   ОШИБКА при чтении/разборе файла: {e}")
    exit()

if not raw_dataset:
    print("   ОШИБКА: Датасет пуст.")
    exit()

print(f"\n3. Подготовка тестовых примеров (Seed={RANDOM_SEED})...")
random.seed(RANDOM_SEED)
random.shuffle(raw_dataset)
test_examples = raw_dataset[:min(NUM_EXAMPLES_TO_TEST, len(raw_dataset))]

print(f"   Будет протестировано {len(test_examples)} примеров.")

print(f"\n4. Тестирование модели '{FINE_TUNED_MODEL_NAME}'...")
try:
    for i, example in enumerate(test_examples):
        if not (isinstance(example, list) and len(example) == 2 and isinstance(example[0], str) and isinstance(example[1], str)):
            print(f"\n   Пропуск некорректного примера {i+1}: {example}")
            continue

        input_text, expected_output = example
        print(f"\n   Пример {i+1}/{len(test_examples)}:")
        print(f"     Вход: '{input_text}'")
        print(f"     Ожидаемый выход: '{expected_output}'")

        try:
            response = client.models.generate_content(
                model=FINE_TUNED_MODEL_NAME,
                contents=[input_text] # Важно передать как список строк
            )

            generated_text = "Не удалось извлечь текст из ответа."
            if response and hasattr(response, 'text'):
                generated_text = response.text
            elif response and hasattr(response, 'parts') and response.parts:
                 text_parts = [part.text for part in response.parts if hasattr(part, 'text')]
                 generated_text = " ".join(text_parts)
            elif response and hasattr(response, 'candidates') and response.candidates:
                 # Обработка структуры с candidates (более новая)
                 if response.candidates[0].content and response.candidates[0].content.parts:
                     generated_text = "".join(part.text for part in response.candidates[0].content.parts)


            print(f"     Сгенерированный выход: '{generated_text}'")

        except core_exceptions.GoogleAPIError as api_err:
            print(f"     ОШИБКА API при генерации для примера {i+1}: {api_err}")
        except Exception as gen_err:
            print(f"     НЕПРЕДВИДЕННАЯ ОШИБКА при генерации для примера {i+1}: {gen_err}")


except Exception as e:
    print(f"\n   Критическая ошибка во время тестирования: {e}")

print("\n--- Скрипт Оценки Модели завершен ---")