from google import genai
from google.api_core import exceptions as core_exceptions
import time

MODEL_TO_DELETE = "tunedModels/youtubecommentgeneratorv2-v2a365188e28" 

print("--- Запуск скрипта Удаления Модели ---")


print(f"1. Создание клиента Gemini...")
try:
    client = genai.Client()
    client.models.list() # Простая проверка аутентификации
    print("   Клиент успешно создан и аутентифицирован.")
except Exception as e:
    print(f"   Ошибка при создании клиента: {e}")
    exit()

print(f"\n2. Попытка удаления модели '{MODEL_TO_DELETE}'...")
try:
    print(f"   Отправка запроса на удаление...")
    # Используем client.models.delete, т.к. работаем с ресурсом 'tunedModels/...'
    client.models.delete(model=MODEL_TO_DELETE)
    print(f"   Запрос на удаление модели '{MODEL_TO_DELETE}' успешно отправлен.")
    print(f"   Примечание: Удаление может занять некоторое время.")

except core_exceptions.NotFound:
    print(f"   ОШИБКА: Модель с именем '{MODEL_TO_DELETE}' не найдена.")
except core_exceptions.GoogleAPIError as e:
    print(f"   ОШИБКА API при попытке удаления: {e}")
    if hasattr(e, 'message'): print(f"   Сообщение: {e.message}")
except Exception as e:
    print(f"   НЕПРЕДВИДЕННАЯ ОШИБКА при попытке удаления: {e}")

print("\n--- Скрипт Удаления Модели завершен ---")