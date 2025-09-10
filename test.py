from google import genai
import google.auth
from colorama import Fore, Style, init
import time
import os
import sys
import reprlib
import datetime

init(autoreset=True)

TUNING_JOB_NAME = "tunedModels/youtubecommentgeneratorv2-r5j5ulp5vft1"

def format_datetime(dt_object):
    if not dt_object or not isinstance(dt_object, datetime.datetime):
        return "N/A"
    try:
        if dt_object.tzinfo:
            dt_object = dt_object.astimezone(datetime.timezone.utc)
            return dt_object.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
             return dt_object.strftime("%Y-%m-%d %H:%M:%S (naive)")
    except Exception:
        return "Invalid Timestamp"

print(f"{Style.BRIGHT}{Fore.CYAN}--- Проверка статуса задачи Fine-Tuning Gemini ---{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}Имя задачи:{Style.RESET_ALL} {TUNING_JOB_NAME}")
print("-" * 50)

try:
    print(f"{Fore.CYAN}1. Создание клиента Gemini...{Style.RESET_ALL}")
    client = genai.Client()
    client.models.list()
    print(f"{Fore.GREEN}   Клиент создан и аутентифицирован.{Style.RESET_ALL}")
except google.auth.exceptions.DefaultCredentialsError as e:
     print(f"{Fore.RED}   ОШИБКА: Не удалось найти учетные данные. Установите GOOGLE_API_KEY.{Style.RESET_ALL}")
     print(f"   Детали: {e}")
     exit()
except Exception as e:
    print(f"{Fore.RED}   Непредвиденная ошибка при создании клиента: {e}{Style.RESET_ALL}")
    exit()

retrieved_object = None
try:
    print(f"\n{Fore.CYAN}2. Запрос информации для задачи...{Style.RESET_ALL}")
    sys.stdout.flush()
    retrieved_object = client.tunings.get(name=TUNING_JOB_NAME)
    print(f"{Fore.GREEN}   Информация получена.{Style.RESET_ALL}")

except Exception as e:
    print(f"{Fore.RED}   ОШИБКА при запросе информации: {e}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   Убедись, что имя '{TUNING_JOB_NAME}' корректно.{Style.RESET_ALL}")
    exit()

print(f"\n{Style.BRIGHT}{Fore.YELLOW}--- Полная информация об объекте (repr) ---{Style.RESET_ALL}")
pretty_repr = reprlib.Repr()
pretty_repr.maxlevel = 60
pretty_repr.maxstring = 1500
pretty_repr.maxother = 1500
print(pretty_repr.repr(retrieved_object))
print(f"{Style.BRIGHT}{Fore.YELLOW}--- Конец полной информации ---{Style.RESET_ALL}")
print("-" * 50)

print(f"\n{Style.BRIGHT}{Fore.CYAN}--- Извлеченные данные ---{Style.RESET_ALL}")

name = getattr(retrieved_object, 'name', 'N/A')
display_name = getattr(retrieved_object, 'tuned_model_display_name', None)
if not display_name and name != 'N/A':
     try:
         display_name = name.split('/')[-1].split('-')[0] + " (derived)"
     except:
         display_name = 'N/A'


description = getattr(retrieved_object, 'description', 'N/A')
current_state_enum = getattr(retrieved_object, 'state', None)
create_time_dt = getattr(retrieved_object, 'create_time', None)
update_time_dt = getattr(retrieved_object, 'update_time', None)
start_time_dt = getattr(retrieved_object, 'start_time', None)
end_time_dt = getattr(retrieved_object, 'end_time', None)
base_model = getattr(retrieved_object, 'base_model', 'N/A')

tuned_model_info = getattr(retrieved_object, 'tuned_model', None)
final_model_name = getattr(tuned_model_info, 'model', None) if tuned_model_info else None
final_model_endpoint = getattr(tuned_model_info, 'endpoint', None) if tuned_model_info else None

error_details = getattr(retrieved_object, 'error', None)


create_time_str = format_datetime(create_time_dt)
update_time_str = format_datetime(update_time_dt)
start_time_str = format_datetime(start_time_dt)
end_time_str = format_datetime(end_time_dt)

print(f"{Fore.MAGENTA}Полное имя задачи: {Style.RESET_ALL} {name}")
print(f"{Fore.MAGENTA}Отображаемое имя: {Style.RESET_ALL} {display_name if display_name else '(не задано)'}")
print(f"{Fore.MAGENTA}Описание:         {Style.RESET_ALL} {description if description else '(не задано)'}")
print(f"{Fore.MAGENTA}Базовая модель:    {Style.RESET_ALL} {base_model}")
print(f"{Fore.MAGENTA}Время создания:   {Style.RESET_ALL} {create_time_str}")
print(f"{Fore.MAGENTA}Время старта:     {Style.RESET_ALL} {start_time_str}")
print(f"{Fore.MAGENTA}Время завершения: {Style.RESET_ALL} {end_time_str}")
print(f"{Fore.MAGENTA}Время обновления: {Style.RESET_ALL} {update_time_str}")

print("-" * 50)
print(f"{Style.BRIGHT}{Fore.CYAN}Статус:{Style.RESET_ALL}")

current_state_str = str(current_state_enum) if current_state_enum else "UNKNOWN"
print(f"  {Fore.MAGENTA}Значение Enum:   {Style.RESET_ALL} {current_state_enum}")
print(f"  {Fore.MAGENTA}Строка статуса: {Style.RESET_ALL} {current_state_str}")


if "JOB_STATE_SUCCEEDED" in current_state_str:
    print(f"  {Fore.GREEN}{Style.BRIGHT}УСПЕШНО ЗАВЕРШЕНО{Style.RESET_ALL}")
    if final_model_name:
        print(f"{Fore.MAGENTA}Имя готовой модели:{Style.RESET_ALL} {Style.BRIGHT}{Fore.YELLOW}{final_model_name}{Style.RESET_ALL}")
        if final_model_endpoint:
             print(f"{Fore.MAGENTA}Endpoint модели:   {Style.RESET_ALL} {Style.BRIGHT}{Fore.YELLOW}{final_model_endpoint}{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}>>> Используйте '{final_model_name}' как имя дообученной модели <<<")
    else:
        print(f"{Fore.YELLOW}Предупреждение: Статус 'Успешно', но имя готовой модели не найдено в 'tuned_model.model'.{Style.RESET_ALL}")

elif "JOB_STATE_FAILED" in current_state_str:
    print(f"  {Fore.RED}{Style.BRIGHT}ОШИБКА ОБУЧЕНИЯ{Style.RESET_ALL}")
    if error_details:
         error_message = getattr(error_details, 'message', str(error_details))
         error_code = getattr(error_details, 'code', 'N/A')
         print(f"{Fore.RED}  Код ошибки:  {error_code}{Style.RESET_ALL}")
         print(f"{Fore.RED}  Сообщение:   {error_message}{Style.RESET_ALL}")
    else:
         print(f"{Fore.RED}  Дополнительные детали ошибки не найдены.{Style.RESET_ALL}")

elif "JOB_STATE_CREATING" in current_state_str:
    print(f"  {Fore.YELLOW}СОЗДАНИЕ / ПОДГОТОВКА...{Style.RESET_ALL}")

elif "JOB_STATE_ACTIVE" in current_state_str or "JOB_STATE_RUNNING" in current_state_str:
     print(f"  {Fore.YELLOW}В ПРОЦЕССЕ ОБУЧЕНИЯ...{Style.RESET_ALL}")

elif "JOB_STATE_STOPPING" in current_state_str:
     print(f"  {Fore.YELLOW}ОСТАНОВКА...{Style.RESET_ALL}")

elif "JOB_STATE_STOPPED" in current_state_str or "JOB_STATE_CANCELLED" in current_state_str:
     print(f"  {Fore.YELLOW}ОСТАНОВЛЕНО / ОТМЕНЕНО.{Style.RESET_ALL}")

else:
    print(f"  {Fore.YELLOW}Неизвестный или неопределенный статус: {current_state_str}{Style.RESET_ALL}")


print("\n--- Проверка завершена ---")