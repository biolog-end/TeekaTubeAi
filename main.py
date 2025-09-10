from google import genai
from google.genai import types
import google.auth
import random
import os
import pickle
import json
import time
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify 
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from colorama import Fore, init
import string
import math 
import requests 
import emoji

from flask_session import Session 

# --- Настройки ---
init(autoreset=True, convert=True)
CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
TOKEN_PICKLE_FILE = "token.pickle"
VIDEO_INFO_FILE = "video_info_posted_web.txt"
DATASET_JSON_FILE = "training_dataset.json"
FINE_TUNED_MODEL_NAME = "tunedModels/youtubecommentgeneratorv1-9fcq06wpeo1j"
BASE_MODEL_FOR_SUMMARIZE = 'gemini-1.5-flash-latest' 
FINE_TUNED_MODEL_NAME_TWO = "tunedModels/youtubecommentgeneratorv2-r5j5ulp5vft1"
MAX_HISTORY_PAIRS = 310
GENERATION_LOG_FILE = "last_generation_request.log"
POST_COMMENT_DELAY = 1
USED_IDS_FILE = 'usedId.json' 

def make_human_like_typos(text: str,
                                      substitution_chance: float = 0.01,
                                      transposition_chance: float = 0.005,
                                      skip_chance: float = 0.002) -> str:
    """
    Добавляет в текст человекоподобные опечатки для русского и английского языков,
    СТРОГО ИЗБЕГАЯ замены букв одного языка на буквы другого.

    Args:
        text: Исходный текст.
        substitution_chance: Вероятность замены буквы на соседнюю ПО ТОЙ ЖЕ РАСКЛАДКЕ (на символ).
        transposition_chance: Вероятность перестановки двух соседних букв (на символ).
        skip_chance: Вероятность пропуска (удаления) буквы или цифры (на символ).

    Returns:
        Текст с возможными опечатками.
    """

    RU_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    RU_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    RU_ALPHABET = set(RU_LOWER + RU_UPPER)

    EN_LOWER = string.ascii_lowercase
    EN_UPPER = string.ascii_uppercase
    EN_ALPHABET = set(EN_LOWER + EN_UPPER)

    RU_TYPO_SUBSTITUTIONS = {
        '.': ['ж', 'ю'],
        '\\': ['ъ'],
        'А': ['К', 'М', 'В', 'П'],
        'Б': ['Л', 'Ь', 'Ю'],
        'В': ['У', 'С', 'Ы', 'А'],
        'Г': ['О', 'Н', 'Ш'],
        'Д': ['Щ', 'Ю', 'Л', 'Ж'],
        'Е': ['П', 'К', 'Н'],
        'Ж': ['З', '.', 'Д', 'Э'],
        'З': ['Ж', 'Щ', 'Х'],
        'И': ['П', 'М', 'Т'],
        'Й': ['Ф', 'Ц'],
        'К': ['А', 'У', 'Е'],
        'Л': ['Ш', 'Б', 'О', 'Д'],
        'М': ['А', 'С', 'И'],
        'Н': ['Р', 'Е', 'Г'],
        'О': ['Г', 'Ь', 'Р', 'Л'],
        'П': ['Е', 'И', 'А', 'Р'],
        'Р': ['Н', 'Т', 'П', 'О'],
        'С': ['В', 'Ч', 'М'],
        'Т': ['Р', 'И', 'Ь'],
        'У': ['В', 'Ц', 'К'],
        'Ф': ['Й', 'Я', 'Ы'],
        'Х': ['Э', 'З', 'Ъ'],
        'Ц': ['Ы', 'Й', 'У'],
        'Ч': ['Ы', 'Я', 'С'],
        'Ш': ['Л', 'Г', 'Щ'],
        'Щ': ['Д', 'Ш', 'З'],
        'Ъ': ['Х', '\\'],
        'Ы': ['Ц', 'Ч', 'Ф', 'В'],
        'Ь': ['О', 'Т', 'Б'],
        'Э': ['Х', 'Ж'],
        'Ю': ['Д', 'Б', '.'],
        'Я': ['Ф', 'Ч'],
        'а': ['к', 'м', 'в', 'п'],
        'б': ['л', 'ь', 'ю'],
        'в': ['у', 'с', 'ы', 'а'],
        'г': ['о', 'н', 'ш'],
        'д': ['щ', 'ю', 'л', 'ж'],
        'е': ['п', 'к', 'н'],
        'ж': ['з', '.', 'д', 'э'],
        'з': ['ж', 'щ', 'х'],
        'и': ['п', 'м', 'т'],
        'й': ['ф', 'ц'],
        'к': ['а', 'у', 'е'],
        'л': ['ш', 'б', 'о', 'д'],
        'м': ['а', 'с', 'и'],
        'н': ['р', 'е', 'г'],
        'о': ['г', 'ь', 'р', 'л'],
        'п': ['е', 'и', 'а', 'р'],
        'р': ['н', 'т', 'п', 'о'],
        'с': ['в', 'ч', 'м'],
        'т': ['р', 'и', 'ь'],
        'у': ['в', 'ц', 'к'],
        'ф': ['й', 'я', 'ы'],
        'х': ['э', 'з', 'ъ'],
        'ц': ['ы', 'й', 'у'],
        'ч': ['ы', 'я', 'с'],
        'ш': ['л', 'г', 'щ'],
        'щ': ['д', 'ш', 'з'],
        'ъ': ['х', '\\'],
        'ы': ['ц', 'ч', 'ф', 'в'],
        'ь': ['о', 'т', 'б'],
        'э': ['х', 'ж'],
        'ю': ['д', 'б', '.'],
        'я': ['ф', 'ч']
    }

    EN_TYPO_SUBSTITUTIONS = {
        'A': ['Q', 'Z', 'S'],
        'B': ['G', 'V', 'N'],
        'C': ['D', 'X', 'V'],
        'D': ['E', 'C', 'S', 'F'],
        'E': ['D', 'W', 'R'],
        'F': ['R', 'V', 'D', 'G'],
        'G': ['T', 'B', 'F', 'H'],
        'H': ['Y', 'N', 'G', 'J'],
        'I': ['K', 'U', 'O'],
        'J': ['U', 'M', 'H', 'K'],
        'K': ['I', ',', 'J', 'L'],
        'L': ['O', '.', 'K', ';'],
        'M': ['J', 'N', ','],
        'N': ['H', 'B', 'M'],
        'O': ['L', 'I', 'P'],
        'P': [';', 'O', '['],
        'Q': ['A', 'W'],
        'R': ['F', 'E', 'T'],
        'S': ['W', 'X', 'A', 'D'],
        'T': ['G', 'R', 'Y'],
        'U': ['J', 'Y', 'I'],
        'V': ['F', 'C', 'B'],
        'W': ['S', 'Q', 'E'],
        'X': ['S', 'Z', 'C'],
        'Y': ['H', 'T', 'U'],
        'Z': ['A', 'X'],
        'a': ['q', 'z', 's'],
        'b': ['g', 'v', 'n'],
        'c': ['d', 'x', 'v'],
        'd': ['e', 'c', 's', 'f'],
        'e': ['d', 'w', 'r'],
        'f': ['r', 'v', 'd', 'g'],
        'g': ['t', 'b', 'f', 'h'],
        'h': ['y', 'n', 'g', 'j'],
        'i': ['k', 'u', 'o'],
        'j': ['u', 'm', 'h', 'k'],
        'k': ['i', ',', 'j', 'l'],
        'l': ['o', '.', 'k', ';'],
        'm': ['j', 'n', ','],
        'n': ['h', 'b', 'm'],
        'o': ['l', 'i', 'p'],
        'p': [';', 'o', '['],
        'q': ['a', 'w'],
        'r': ['f', 'e', 't'],
        's': ['w', 'x', 'a', 'd'],
        't': ['g', 'r', 'y'],
        'u': ['j', 'y', 'i'],
        'v': ['f', 'c', 'b'],
        'w': ['s', 'q', 'e'],
        'x': ['s', 'z', 'c'],
        'y': ['h', 't', 'u'],
        'z': ['a', 'x']
    }

    RU_TYPO_TRANSPOSITIONS = [
        "ст", "тс", "ол", "ло", "ть", "ьт", "но", "он", "ер", "ре",
        "ов", "во", "пр", "рп", "на", "ан", "ко", "ок", "то", "от", "ет", "те",
        "СТ", "ТС", "ОЛ", "ЛО", "ТЬ", "ЬТ", "НО", "ОН", "ЕР", "РЕ",
        "ОВ", "ВО", "ПР", "РП", "НА", "АН", "КО", "ОК", "ТО", "ОТ", "ЕТ", "ТЕ"
    ]

    EN_TYPO_TRANSPOSITIONS = [
        "on", "no", "re", "er", "th", "ht", "in", "ni", "at", "ta",
        "en", "ne", "es", "se", "ou", "uo", "is", "si", "of", "fo", "he", "eh",
        "ON", "NO", "RE", "ER", "TH", "HT", "IN", "NI", "AT", "TA",
        "EN", "NE", "ES", "SE", "OU", "UO", "IS", "SI", "OF", "FO", "HE", "EH"
    ]

    ALL_TYPO_TRANSPOSITIONS_SET = set(RU_TYPO_TRANSPOSITIONS + EN_TYPO_TRANSPOSITIONS)


    chars = list(text)
    new_chars = []
    i = 0
    while i < len(chars):
        char = chars[i]
        processed = False 

        
        if i + 1 < len(chars):
            next_char = chars[i+1]
            current_pair = char + next_char
            
            is_letter_pair = char.isalpha() and next_char.isalpha()
            same_language = (char in RU_ALPHABET and next_char in RU_ALPHABET) or \
                            (char in EN_ALPHABET and next_char in EN_ALPHABET)

          
            should_check_transposition = (current_pair in ALL_TYPO_TRANSPOSITIONS_SET) or \
                                         (is_letter_pair and same_language) or \
                                         (char.isdigit() and next_char.isdigit()) 

            if should_check_transposition and random.random() < transposition_chance:
                new_chars.append(next_char)
                new_chars.append(char)
                i += 2 
                processed = True

        if not processed:

            if char.isalnum() and random.random() < skip_chance:
                i += 1
                processed = True 

            
            elif random.random() < substitution_chance:
                substituted = False
                
                if char in RU_ALPHABET and char in RU_TYPO_SUBSTITUTIONS:
                    possible_typos = RU_TYPO_SUBSTITUTIONS[char]
                    if possible_typos: 
                        typo_char = random.choice(possible_typos)
                        new_chars.append(typo_char)
                        substituted = True
                
                elif char in EN_ALPHABET and char in EN_TYPO_SUBSTITUTIONS:
                     possible_typos = EN_TYPO_SUBSTITUTIONS[char]
                     if possible_typos:
                        typo_char = random.choice(possible_typos)
                        new_chars.append(typo_char)
                        substituted = True
             
                elif (not char.isalnum()) and char in RU_TYPO_SUBSTITUTIONS:
                    possible_typos = RU_TYPO_SUBSTITUTIONS[char]
                    if possible_typos:
                        typo_char = random.choice(possible_typos)
                        new_chars.append(typo_char)
                        substituted = True
                 
                elif (not char.isalnum()) and char in EN_TYPO_SUBSTITUTIONS:
                    possible_typos = EN_TYPO_SUBSTITUTIONS[char]
                    if possible_typos:
                        typo_char = random.choice(possible_typos)
                        new_chars.append(typo_char)
                        substituted = True

                
                if substituted:
                    i += 1
                    processed = True

        
        if not processed:
            new_chars.append(char)
            i += 1

    return "".join(new_chars)


def authenticate_and_get_credentials():
    """Проверяет, получает или обновляет учетные данные в КОНСОЛИ."""
    credentials = None
    if os.path.exists(TOKEN_PICKLE_FILE):
        try:
            with open(TOKEN_PICKLE_FILE, "rb") as token:
                credentials = pickle.load(token)
            print(Fore.GREEN + f"Учетные данные загружены из {TOKEN_PICKLE_FILE}.")
        except (EOFError, pickle.UnpicklingError):
             print(Fore.YELLOW + f"Ошибка чтения {TOKEN_PICKLE_FILE}. Потребуется новая аутентификация.")
             credentials = None
             if os.path.exists(TOKEN_PICKLE_FILE):
                 os.remove(TOKEN_PICKLE_FILE)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            print(Fore.YELLOW + "Срок действия токена истек, пытаюсь обновить...")
            try:
                credentials.refresh(Request())
                print(Fore.GREEN + "Токен успешно обновлен.")
                with open(TOKEN_PICKLE_FILE, "wb") as token:
                    pickle.dump(credentials, token)
                print(Fore.BLUE + f"Обновленные учетные данные сохранены в {TOKEN_PICKLE_FILE}.")
            except Exception as e:
                print(Fore.RED + f"Не удалось обновить токен: {e}")
                print(Fore.YELLOW + f"Потребуется ручная аутентификация. Удалите {TOKEN_PICKLE_FILE}, если проблемы сохраняются.")
                credentials = None
        else:
             if not os.path.exists(CLIENT_SECRETS_FILE):
                 print(Fore.RED + f"ОШИБКА: Файл секрета клиента '{CLIENT_SECRETS_FILE}' не найден!")
                 print(Fore.YELLOW + "Пожалуйста, скачайте его из Google Cloud Console и поместите в папку с программой.")
                 return None

             print(Fore.CYAN + "Запуск процесса аутентификации в браузере...")
             flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
             credentials = flow.run_local_server(port=0)
             with open(TOKEN_PICKLE_FILE, "wb") as token:
                 pickle.dump(credentials, token)
             print(Fore.GREEN + "Аутентификация прошла успешно.")
             print(Fore.BLUE + f"Учетные данные сохранены в {TOKEN_PICKLE_FILE}.")

    if credentials and credentials.valid:
        return credentials
    else:
        print(Fore.RED + "Не удалось получить действительные учетные данные.")
        return None


# --- Глобальные переменные ---
app = Flask(__name__)
app.secret_key = 'super secret key teekas'
g_credentials = None
gemini_client = None 
# --- Настройка Flask-Session ---
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_COOKIE_SECURE'] = False 
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax' 

server_session = Session(app)

# --- Функции Google API ---
def get_youtube_service():
    """Создает сервис YouTube, используя ГЛОБАЛЬНЫЕ учетные данные."""
    global g_credentials
    if not g_credentials or not g_credentials.valid:
        if g_credentials and g_credentials.expired and g_credentials.refresh_token:
             try:
                 print("Обновление токена во время выполнения Flask...")
                 g_credentials.refresh(Request())
                 with open(TOKEN_PICKLE_FILE, "wb") as token:
                    pickle.dump(g_credentials, token)
             except Exception as e:
                 flash(f"Критическая ошибка: Не удалось обновить токен во время работы: {e}", "error")
                 print(Fore.RED + f"Не удалось обновить токен во время работы: {e}")
                 return None
        else:
            flash("Критическая ошибка: Отсутствуют или недействительны учетные данные.", "error")
            print(Fore.RED + "Отсутствуют или недействительны учетные данные в get_youtube_service.")
            return None
    try:
        youtube = build('youtube', 'v3', credentials=g_credentials)
        return youtube
    except HttpError as e:
        flash(f"Ошибка при создании сервиса YouTube: {e}", "error")
        return None
    except Exception as e:
        flash(f"Неизвестная ошибка при создании сервиса YouTube: {e}", "error")
        return None

def get_related_videos_by_search(youtube, query, max_results=5):
    """Ищет видео по запросу и фильтрует результаты, исключая уже использованные ID."""
    if not youtube: return []

    try:
        print(f"Поиск связанных видео по запросу: '{query[:50]}...' (max: {max_results})")
        request = youtube.search().list(
            part="snippet",
            type="video",
            maxResults=max_results,
            q=query
        )
        response = request.execute()
        items = response.get('items', [])

        if not items:
             print("Поиск не дал результатов.")
             return []

        used_ids = load_used_ids()
        if not used_ids:
            print("Нет использованных ID для фильтрации связанных видео.")
            return items

        print(f"Загружено {len(used_ids)} использованных ID для фильтрации результатов поиска.")
        filtered_items = []
        removed_count = 0
        for item in items:
            video_id = item.get('id', {}).get('videoId')
            if video_id and video_id not in used_ids:
                filtered_items.append(item)
            elif video_id:
                 removed_count += 1

        if removed_count > 0:
             print(f"Отфильтровано {removed_count} связанных видео, так как они уже обработаны.")
        print(f"Возвращаем {len(filtered_items)} связанных видео после фильтрации.")

        return filtered_items

    except HttpError as e:
        print(Fore.RED + f"Ошибка API при поиске связанных видео ({query[:50]}...): {e}")
        return []
    except Exception as e:
        print(Fore.RED + f"Ошибка при поиске связанных видео: {e}")
        return []


def get_video_details(youtube, video_id):
    if not youtube: return None
    try:
        video_request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        video_response = video_request.execute()
        if not video_response.get('items'):
            flash(f"Детали для видео ID {video_id} не найдены.", "warning")
            return None

        item = video_response['items'][0]
        video_info = item.get('snippet', {})

        title = video_info.get('title', 'Без названия')
        description = video_info.get('description', 'Без описания') 
        channel_title = video_info.get('channelTitle', 'Без названия канала')
        tags = video_info.get('tags', [])

        comments_data = []
        try:
            comment_request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                order="relevance", 
                maxResults=3
            )
            comment_response = comment_request.execute()
            for item in comment_response.get('items', []):
                 try:
                     top_comment = item.get('snippet', {}).get('topLevelComment', {})
                     comment_snippet = top_comment.get('snippet', {})
                     author = comment_snippet.get('authorDisplayName', 'Неизвестный автор')
                     text = comment_snippet.get('textDisplay', '')
                     comment_id = top_comment.get('id', '')
                     comments_data.append({
                         'author': author, 
                         'text': text, 
                         'id': comment_id
                     })
                 except AttributeError: 
                     comments_data.append({
                         'author': 'Ошибка данных', 
                         'text': 'Не удалось извлечь коммент',
                         'id': ''
                     })

        except HttpError as e:
            if e.resp.status == 403:
                 flash(f"Комментарии для видео {video_id} отключены или недоступны.", "info")
                 print(Fore.YELLOW + f"Комментарии для видео {video_id} отключены или недоступны.")
            else:
                 flash(f"Ошибка API при получении комментариев для {video_id}: {e}", "warning")
                 print(Fore.YELLOW + f"Ошибка при получении комментариев для {video_id}: {e}")
        except Exception as e:
            flash(f"Неизвестная ошибка при получении комментариев для {video_id}: {e}", "warning")
            print(Fore.YELLOW + f"Неизвестная ошибка при получении комментариев для {video_id}: {e}")

        return title, channel_title, description, tags, comments_data

    except HttpError as e:
        flash(f"Ошибка API при получении деталей видео {video_id}: {e}", "error")
        print(Fore.RED + f"Ошибка API при получении деталей видео {video_id}: {e}")
        return None
    except Exception as e:
        flash(f"Ошибка при получении деталей видео {video_id}: {e}", "error")
        print(Fore.RED + f"Ошибка при получении деталей видео {video_id}: {e}")
        return None
    
def add_used_id(video_id):
    """Добавляет один video_id в файл usedId.json, сохраняя структуру {"ids": [...]}. Потокобезопасно."""
    if not video_id: 
        return False

    used_ids = load_used_ids() 

    if video_id in used_ids:
        return False 

    used_ids.add(video_id) 

    try:
        ids_list = sorted(list(used_ids))
        with open(USED_IDS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"ids": ids_list}, f, indent=4) 
        print(f"ID {video_id} успешно добавлен в {USED_IDS_FILE}")
        return True 
    except IOError as e:
        print(f"Ошибка записи в файл {USED_IDS_FILE}: {e}")
        return False
    except Exception as e:
            print(f"Неизвестная ошибка при сохранении файла {USED_IDS_FILE}: {e}")
            return False

def post_comment(youtube, video_id, comment_text, reply_to_id=None):
    if not youtube: return False
    if not comment_text:
        flash("Текст комментария пуст.", "warning")
        return False
    
    try:
        if reply_to_id:
            request = youtube.comments().insert(
                part="snippet",
                body={
                    "snippet": {
                        "parentId": reply_to_id,
                        "textOriginal": comment_text
                    }
                }
            )
            response = request.execute()
            flash("Ответ на комментарий успешно опубликован!", "success")
            print(Fore.GREEN + f"Ответ на комментарий {reply_to_id} опубликован.")
        else:
            request = youtube.commentThreads().insert(
                part="snippet",
                body={
                    "snippet": {
                        "videoId": video_id,
                        "topLevelComment": {
                            "snippet": {
                                "textOriginal": comment_text
                            }
                        }
                    }
                }
            )
            response = request.execute()
            flash("Комментарий успешно опубликован!", "success")
            print(Fore.GREEN + f"Комментарий к {video_id} опубликован.")
            
        add_used_id(video_id)
        return True
    except HttpError as e:
        flash(f"Ошибка API при публикации комментария к {video_id}: {e}", "error")
        print(Fore.RED + f"Ошибка API при публикации комментария к {video_id}: {e}")

def save_video_info(video_id, title, channel_title, description, comments_data, comment_text, replied_to=None, summarized_description=None): # Добавлен параметр summarized_description
    # --- Сохранение в TXT файл ---
    try:
        with open(VIDEO_INFO_FILE, "a", encoding="utf-8") as file:
            file.write(f"Video ID: {video_id}\n")
            file.write(f"Video name: {title}\n")
            file.write(f"Channel name: {channel_title}\n")
            file.write("=" * 20 + " Description " + "=" * 20 + "\n")
            file.write(f"{description}\n") 
            file.write("=" * 20 + " Popular Comments " + "=" * 20 + "\n")
            for i, c_data in enumerate(comments_data[:3]):
                author = c_data.get('author', 'N/A')
                text = c_data.get('text', 'N/A')
                comment_id = c_data.get('id', 'N/A')
                file.write(f"Comment {i+1} (ID: {comment_id}) by [{author}]:\n{text}\n---\n")
            for i in range(len(comments_data), 3):
                 file.write(f"Comment {i+1}: N/A\n---\n")

            file.write("=" * 20 + " Posted Comment " + "=" * 20 + "\n")

            if replied_to:
                replied_comment = None
                replied_author = "Unknown"

                for c_data in comments_data:
                    if c_data.get('id') == replied_to:
                        replied_comment = c_data.get('text', 'N/A')
                        replied_author = c_data.get('author', 'N/A')
                        break

                file.write(f"REPLY TO COMMENT BY [{replied_author}]:\n")
                file.write(f"Original comment: {replied_comment}\n")
                file.write(f"Reply: {comment_text}\n")
            else:
                file.write(f"{comment_text}\n")

            file.write("=" * 80 + "\n\n")
        print(Fore.BLUE + f"Информация о {video_id} сохранена в {VIDEO_INFO_FILE}")
    except Exception as e:
        flash(f"Ошибка сохранения информации в TXT файл для {video_id}: {e}", "error")
        print(Fore.RED + f"Ошибка сохранения информации в TXT файл для {video_id}: {e}")

    # --- Сохранение/Обновление JSON датасета ---
    dataset = []
    if os.path.exists(DATASET_JSON_FILE):
        try:
            with open(DATASET_JSON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "dataset" in data and isinstance(data["dataset"], list):
                    dataset = data["dataset"]
                else:
                    print(Fore.YELLOW + f"Предупреждение: Файл {DATASET_JSON_FILE} имеет неверную структуру. Создаю новый список.")
        except json.JSONDecodeError:
            print(Fore.YELLOW + f"Предупреждение: Не удалось декодировать JSON из {DATASET_JSON_FILE}. Создаю новый список.")
        except Exception as e:
            print(Fore.YELLOW + f"Предупреждение: Не удалось прочитать {DATASET_JSON_FILE}: {e}. Создаю новый список.")

    description_for_json = summarized_description if summarized_description else description

    info_parts = [
        f"Title: {title}",
        f"Channel: {channel_title}",
        f"Description:\n{description_for_json}",
        "--- Comments ---"
    ]
    for i, c_data in enumerate(comments_data[:3]):
         author = c_data.get('author', 'N/A')
         text = c_data.get('text', 'N/A')
         comment_id = c_data.get('id', 'N/A')
         info_parts.append(f"Comment {i+1} by {author}:\n{text}")
    for i in range(len(comments_data), 3):
        info_parts.append(f"Comment {i+1}: N/A")

    video_info_string = "\n".join(info_parts)

    if replied_to:
        replied_comment_index = None
        for i, c_data in enumerate(comments_data):
            if c_data.get('id') == replied_to:
                replied_comment_index = i + 1
                break

        reply_prefix = f"ответить комментарий {replied_comment_index}: " if replied_comment_index else ""
        formatted_comment = f"{reply_prefix}{comment_text}"
        new_data_pair = [video_info_string, formatted_comment]
    else:
        new_data_pair = [video_info_string, comment_text]

    dataset.append(new_data_pair)

    try:
        with open(DATASET_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump({"dataset": dataset}, f, indent=4, ensure_ascii=False)
        print(Fore.BLUE + f"Данные добавлены в датасет {DATASET_JSON_FILE}")
    except Exception as e:
        flash(f"Ошибка сохранения датасета в JSON файл для {video_id}: {e}", "error")
        print(Fore.RED + f"Ошибка сохранения датасета в JSON файл для {video_id}: {e}")

def final_fine_tune_comment(comment: str,                     
                    substitution_chance=0.004,
                    transposition_chance=0.009,
                    skip_chance=0.002) -> str:
    """
    Очищает и форматирует комментарий от нейросети согласно правилам:
    1. Заменяет '"' на '"'.
    2. Удаляет точку в конце, если она не является частью многоточия (...),
       даже если за ней следуют пробелы или эмодзи.

    Args:
        comment: Исходный строковый комментарий.

    Returns:
        Очищенный строковый комментарий.
    """
    if not comment:
        return "" 

    cleaned_comment = comment.replace('&quot;', '"')

    n = len(cleaned_comment)
    last_significant_char_index = -1

    for i in range(n - 1, -1, -1):
        char = cleaned_comment[i]
        if not char.isspace() and not emoji.is_emoji(char):
            last_significant_char_index = i
            break 

    if last_significant_char_index != -1 and cleaned_comment[last_significant_char_index] == '.':
        if last_significant_char_index == 0 or cleaned_comment[last_significant_char_index - 1] != '.':
            cleaned_comment = cleaned_comment[:last_significant_char_index] + cleaned_comment[last_significant_char_index + 1:]

    return make_human_like_typos(cleaned_comment, substitution_chance, transposition_chance, skip_chance)

def generate_tuned_comment(gemini_client, model_name, system_prompt, data, config=None):
    global DATASET_JSON_FILE, MAX_HISTORY_PAIRS, GENERATION_LOG_FILE

    if not gemini_client:
        return jsonify({"error": "Клиент Gemini не инициализирован."}), 500

    title = data.get('title')
    channel = data.get('channel')
    description = data.get('description')
    comments_list = data.get('comments', [])
    user_prompt = data.get('prompt')
    use_history = data.get('use_history', False)

    apply_human_typos = data.get('human_typos', False)

    if not title or not channel or description is None:
        return jsonify({"error": "Не предоставлены название, канал или описание."}), 400

    is_tuned_model = model_name.startswith("tunedModels/")

    comments_str_parts = []
    for i, c_data in enumerate(comments_list[:3]):
        author = c_data.get('author', 'N/A')
        text = c_data.get('text', 'N/A')
        comments_str_parts.append(f"Comment {i+1} by {author}:\n{text}")
    for i in range(len(comments_list), 3):
        comments_str_parts.append(f"Comment {i+1}: N/A")
    comments_block = "\n".join(comments_str_parts)

    current_video_info_string = f"\n\nTitle: {title}\nChannel: {channel}\nDescription:\n{description}\n--- Comments ---\n{comments_block}\n"

    contents_list = []
    user_content_base = ""
    system_instruction_to_pass = None

    if user_prompt:
        user_content_base = user_prompt
        print(Fore.YELLOW + "Используется пользовательский промпт, информация о видео будет передана только в истории (если включена).")
    else:
        user_content_base = current_video_info_string

    final_user_content_text = user_content_base

    if system_prompt:
        if is_tuned_model:
            final_user_content_text = f"{system_prompt}\n{user_content_base}"
            print(Fore.CYAN + "Модель тюнингованная: системный промпт добавлен в текст пользователя.")
        else:
            system_instruction_to_pass = system_prompt
            print(Fore.CYAN + "Модель обычная: системный промпт будет передан как system_instruction.")

    history_loaded = False
    limited_history = []

    if use_history and MAX_HISTORY_PAIRS > 0:
        print(Fore.CYAN + f"Активирован режим истории (макс. {MAX_HISTORY_PAIRS} пар). Загрузка данных...")
        try:
            if os.path.exists(DATASET_JSON_FILE):
                with open(DATASET_JSON_FILE, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    if isinstance(history_data, dict) and "dataset" in history_data and isinstance(history_data["dataset"], list):
                        dataset_history = history_data["dataset"]
                        if dataset_history:
                            limited_history = dataset_history[-MAX_HISTORY_PAIRS:]
                            print(Fore.BLUE + f"Загружено {len(dataset_history)} пар сообщений. Используется последних {len(limited_history)} для контекста.")

                            for i, pair in enumerate(limited_history):
                                if len(pair) == 2 and isinstance(pair[0], str) and isinstance(pair[1], str):
                                    contents_list.append(types.Content(
                                        role="user",
                                        parts=[types.Part.from_text(text=pair[0])]
                                    ))
                                    contents_list.append(types.Content(
                                        role="model",
                                        parts=[types.Part.from_text(text=pair[1])]
                                    ))
                                else:
                                    print(Fore.YELLOW + f"Предупреждение: Некорректный формат пары #{i+1} в истории {DATASET_JSON_FILE}. Пропускаю.")

                            contents_list.append(types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=final_user_content_text)]
                            ))
                            history_loaded = True
                            print(Fore.CYAN + "История (ограниченная) успешно сформирована для multi-turn запроса.")
                        else:
                             print(Fore.YELLOW + f"Предупреждение: Файл {DATASET_JSON_FILE} содержит пустой 'dataset'. История не будет использована.")
                    else:
                        print(Fore.YELLOW + f"Предупреждение: Файл {DATASET_JSON_FILE} имеет неверную структуру. История не будет использована.")
            else:
                print(Fore.YELLOW + f"Предупреждение: Файл истории {DATASET_JSON_FILE} не найден. История не будет использована.")

        except json.JSONDecodeError:
            print(Fore.RED + f"Ошибка: Не удалось декодировать JSON из {DATASET_JSON_FILE}. История не будет использована.")
            history_loaded = False
        except Exception as e:
            print(Fore.RED + f"Ошибка при обработке файла истории {DATASET_JSON_FILE}: {e}. История не будет использована.")
            history_loaded = False

        if not history_loaded:
            print(Fore.YELLOW + "Генерация будет выполнена без истории из-за ошибки загрузки или пустого файла.")
            contents_list = [types.Content(role="user", parts=[types.Part.from_text(text=final_user_content_text)])]
            use_history = False

    elif use_history and MAX_HISTORY_PAIRS <= 0:
         print(Fore.CYAN + "Режим истории включен, но MAX_HISTORY_PAIRS <= 0. История не будет использована.")
         contents_list = [types.Content(role="user", parts=[types.Part.from_text(text=final_user_content_text)])]
         use_history = False
    else:
        print(Fore.CYAN + "Режим истории отключен. Используется стандартный single-turn запрос.")
        contents_list = [types.Content(role="user", parts=[types.Part.from_text(text=final_user_content_text)])]

    default_config = {}
    final_config = config if config is not None else default_config

    try:
        log_prefix = f"ГЕНЕРАЦИЯ комментария (Видео: '{title}') в Gemini (модель: {model_name})"
        if user_prompt:
            log_prefix = f"ГЕНЕРАЦИЯ комментария с пользовательским промптом в Gemini (модель: {model_name})"
        if use_history and history_loaded:
            log_prefix += f" [С ИСТОРИЕЙ ({len(limited_history)} пар)]"
        elif use_history and not history_loaded:
             log_prefix += " [ИСТОРИЯ НЕ ЗАГРУЖЕНА]"
        if system_instruction_to_pass:
            log_prefix += " [С system_instruction]"


        print(Fore.MAGENTA + f"Отправка запроса на {log_prefix}...")
        print(Fore.BLUE + f"Используемое имя модели: {model_name}")

        log_content_to_write = ""
        if isinstance(contents_list, list) and contents_list:
             if isinstance(contents_list[0], types.Content):
                  print(Fore.BLUE + f"Отправляется {len(contents_list)} сообщений.")
                  if contents_list[-1].parts:
                      print(Fore.BLUE + f"Последнее сообщение (user):\n---\n{contents_list[-1].parts[0].text[:300]}...\n---")

                  formatted_log_parts = []
                  for item in contents_list:
                      role_prefix = f"--- {item.role.upper()} ---"
                      text_part = ""
                      if item.parts:
                          text_part = "\n".join([part.text for part in item.parts if hasattr(part, 'text')])
                      formatted_log_parts.append(f"{role_prefix}\n{text_part}\n")
                  log_content_to_write = "\n".join(formatted_log_parts)
             else:
                  print(Fore.YELLOW + "Предупреждение: Неожиданный формат элемента в contents_list.")
                  log_content_to_write = f"Ошибка: Неожиданный формат contents_list: {type(contents_list[0])}"
        else:
              print(Fore.RED + "ОШИБКА: Список contents_list пуст перед отправкой запроса!")
              return jsonify({"error": "Внутренняя ошибка: не удалось сформировать контент для запроса."}), 500

        try:
            with open(GENERATION_LOG_FILE, 'w', encoding='utf-8') as log_f:
                log_f.write(f"Model: {model_name}\n")
                log_f.write(f"Config: {json.dumps(final_config)}\n")
                if system_instruction_to_pass:
                    log_f.write("="*30 + " SYSTEM INSTRUCTION " + "="*30 + "\n")
                    log_f.write(system_instruction_to_pass)
                    log_f.write("\n")
                log_f.write("="*30 + " CONTENTS " + "="*30 + "\n")
                log_f.write(log_content_to_write)
                log_f.write(f"Рил запрос (contents) - {contents_list}\n") 
                log_f.write(f"Рил запрос (system_instruction) - {system_instruction_to_pass}\n") 
            print(Fore.LIGHTBLACK_EX + f"Детали запроса записаны в {GENERATION_LOG_FILE}")
        except Exception as log_e:
            print(Fore.YELLOW + f"Предупреждение: Не удалось записать лог запроса в файл {GENERATION_LOG_FILE}: {log_e}")

        api_args = {
            "model": model_name,
            "contents": contents_list,
            "config": final_config
        }
        if system_instruction_to_pass:
            api_args["config"] = types.GenerateContentConfig(system_instruction=[types.Part.from_text(text=system_instruction_to_pass)])

        response = gemini_client.models.generate_content(**api_args)

        generated_comment = None
        reason_empty = "Причина неизвестна"

        if not response.candidates:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                reason_empty = f"Заблокировано Gemini: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name}"
            else:
                reason_empty = "Ответ не содержит кандидатов (возможно, пустой или отфильтрован)."
        elif not hasattr(response, 'text') or not response.text:
             reason_empty = "Ответ содержит кандидатов, но свойство .text пустое или отсутствует."
             if response.candidates:
                 finish_reason = getattr(response.candidates[0], 'finish_reason', None)
                 if finish_reason:
                     reason_empty += f" Причина завершения: {finish_reason.name if hasattr(finish_reason, 'name') else finish_reason}"
        else:
            generated_comment = response.text.strip()

        if generated_comment is None:
            print(Fore.YELLOW + f"Модель '{model_name}' вернула пустой ответ. {reason_empty}")
            return jsonify({"error": f"Модель '{model_name}' не вернула текст. {reason_empty}"}), 500

        print(Fore.GREEN + f"Комментарий успешно сгенерирован моделью '{model_name}'.")
    
        if apply_human_typos:
            print(Fore.CYAN + "Применяем 'человеческие' опечатки...")
            try:
                generated_comment = final_fine_tune_comment(generated_comment)
                print(Fore.CYAN + "Опечатки применены.")
            except NameError:
                print(Fore.RED + "ОШИБКА: Функция 'final_fine_tune_comment' не найдена! Возвращаем комментарий без опечаток.")
            except Exception as typo_error:
                print(Fore.RED + f"ОШИБКА при применении опечаток: {typo_error}. Возвращаем комментарий без опечаток.")

        return jsonify({"comment": generated_comment}), 200

    except Exception as e:
        print(Fore.RED + f"Ошибка при вызове модели '{model_name}': {e}")
        error_message = str(e)
        response_obj_during_exception = None
        if 'response' in locals() or 'response' in globals():
             potential_response = locals().get('response') or globals().get('response')
             if potential_response is not None:
                 response_obj_during_exception = potential_response
                 print(Fore.YELLOW + f"Объект 'response' во время исключения: {response_obj_during_exception}")
                 try:
                    if (hasattr(potential_response, 'prompt_feedback') and
                        potential_response.prompt_feedback is not None and
                        potential_response.prompt_feedback.block_reason):
                           reason_msg = potential_response.prompt_feedback.block_reason_message or potential_response.prompt_feedback.block_reason.name
                           error_message = f"Заблокировано Gemini: {reason_msg}"
                 except Exception as inner_e:
                     print(Fore.YELLOW + f"Дополнительная ошибка при проверке prompt_feedback: {inner_e}")

        try:
            if hasattr(e, 'message') and e.message and error_message == str(e):
                error_message = e.message
        except Exception as inner_e:
             print(Fore.YELLOW + f"Дополнительная ошибка при извлечении e.message: {inner_e}")

        suffix = " (Возможно, модель еще не готова, имя указано неверно, проблемы с API-ключом или формат запроса/истории не поддерживается?)"
        if model_name in str(e) or model_name in error_message:
            if not error_message.endswith(suffix):
                 error_message += suffix

        return jsonify({"error": f"Ошибка модели '{model_name}': {error_message}"}), 500

def load_used_ids():
    """Загружает использованные ID из JSON-файла."""
    if not os.path.exists(USED_IDS_FILE):
        return set() 
    try:
        with open(USED_IDS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'ids' in data and isinstance(data['ids'], list):
                 return set(data['ids'])
            else:
                 print(f"Предупреждение: Неверный формат файла {USED_IDS_FILE}. Ожидался {{'ids': [...]}}.")
                 return set() 
    except json.JSONDecodeError:
        print(f"Ошибка: Не удалось прочитать JSON из файла {USED_IDS_FILE}. Файл может быть поврежден.")
        return set() 
    except Exception as e:
        print(f"Неизвестная ошибка при чтении файла {USED_IDS_FILE}: {e}")
        return set()

# --- Маршруты Flask ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_ids_input = request.form.get('video_ids')

        if video_ids_input:
            video_ids = [vid.strip() for vid in video_ids_input.split(',') if vid.strip()]
            if video_ids:
                session['video_queue'] = video_ids
                session['processed_ids'] = []
                session.pop('current_video_details', None)
                flash(f"Начинаем обработку {len(video_ids)} видео.", "info")
                return redirect(url_for('process_next_video'))
            else:
                flash("Пожалуйста, введите хотя бы один Video ID.", "warning")
        else:
            flash("Поле Video IDs не может быть пустым.", "warning")

    session.pop('video_queue', None)
    session.pop('processed_ids', None)
    session.pop('current_video_details', None)

    used_ids_json = json.dumps(list(load_used_ids())) 

    return render_template('index.html', title="Старт", used_ids_json=used_ids_json)


@app.route('/process')
def process_next_video():
    youtube = get_youtube_service()
    if not youtube:
        return render_template('error.html', title="Ошибка сервиса YouTube")

    if 'video_queue' not in session or not session['video_queue']:
        session.pop('processed_ids', None)
        session.pop('current_video_details', None)
        flash("Очередь видео пуста. Обработка завершена.", "success")
        return render_template('finished.html', title="Завершено")

    processed_ids = set(session.get('processed_ids', []))
    original_queue = session['video_queue']
    next_video_id = None

    for video_id in original_queue:
         if video_id not in processed_ids:
             next_video_id = video_id
             break

    if not next_video_id:
         temp_queue = list(original_queue)
         while temp_queue:
             candidate_id = temp_queue.pop(0)
             if candidate_id not in processed_ids:
                 next_video_id = candidate_id
                 break

    if not next_video_id:
         flash("Все видео в очереди обработаны. Завершение.", "info")
         session.pop('video_queue', None)
         session.pop('processed_ids', None)
         session.pop('current_video_details', None)
         return render_template('finished.html', title="Завершено")

    print(Fore.CYAN + f"\n--- Обработка видео ID: {next_video_id} ---")
    details = get_video_details(youtube, next_video_id) 

    if details:
        title, channel_title, description, tags, comments_data = details
        session['current_video_details'] = {
            'id': next_video_id, 'title': title, 'channel': channel_title,
            'description': description, 'tags': tags, 'comments': comments_data 
        }
        queue_set = set(original_queue)
        remaining_count = len(queue_set - processed_ids) -1

        return render_template('process_video.html',
                               video_id=next_video_id, title=title, channel=channel_title,
                               description=description, 
                               comments=comments_data, 
                               queue_count=max(0, remaining_count),
                               default_related_index=5 
                               )
    else:
        flash(f"Не удалось получить детали для {next_video_id}. Пропускаем.", "warning")
        current_processed = session.get('processed_ids', [])
        if next_video_id not in current_processed:
             current_processed.append(next_video_id)
             session['processed_ids'] = current_processed
        current_queue = session.get('video_queue', [])
        session['video_queue'] = [vid for vid in current_queue if vid != next_video_id]
        session.modified = True
        time.sleep(1)
        return redirect(url_for('process_next_video'))

@app.route('/summarize_description', methods=['POST'])
def summarize_description_route():
    if not gemini_client: 
        return jsonify({"error": "Клиент Gemini не инициализирован."}), 500

    original_description = request.json.get('description')
    video_title = request.json.get('title')
    channel_name = request.json.get('channel')
    if not original_description or not video_title or not channel_name:
        missing = [k for k, v in {'описание': original_description, 'название': video_title, 'канал': channel_name}.items() if not v]
        return jsonify({"error": f"Не предоставлены данные: {', '.join(missing)}."}), 400
    prompt = f"""
    Сократи следующее описание видео с YouTube, учитывая его название и название канала. Оставь только самую суть, удали:
    - Призывы к действию (подписаться, поставить лайк, перейти по ссылкам).
    - Рекламные интеграции или упоминания спонсоров.
    - Ссылки на социальные сети, другие каналы, плейлисты (если они не являются основной темой видео, указанной в названии).
    - Неинформативные или повторяющиеся теги, перечисленные в описании, особенно если они явно не связаны с названием видео или каналом.
    - Приветствия, прощания, общие фразы.
    - Тайм-коды (если они не критичны для понимания сути).

    Сохрани ключевую информацию о содержании видео, как указано в названии и подкреплено описанием. Игнорируй теги, если они противоречат названию видео. Ответ должен быть только текстом сокращенного описания, без твоих комментариев до или после.

    Название видео: {video_title}
    Канал: {channel_name}

    Оригинальное описание:
    ---
    {original_description}
    ---

    Сокращенное описание:
    """

    try:
        print(Fore.MAGENTA + f"Отправка запроса на СОКРАЩЕНИЕ (Видео: '{video_title}') в Gemini (модель: {BASE_MODEL_FOR_SUMMARIZE})...")
        response = gemini_client.models.generate_content(
            model=BASE_MODEL_FOR_SUMMARIZE,
            contents=[prompt]
        )
        if not response.text:
             reason = "Причина неизвестна"
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: reason = f"Заблокировано Gemini: {response.prompt_feedback.block_reason_message}"
             elif hasattr(response, 'candidates') and not response.candidates: reason = "Ответ не содержит кандидатов."
             print(Fore.YELLOW + f"Gemini (summarize) вернул пустой ответ. {reason}")
             return jsonify({"error": f"Gemini (summarize) не вернул текст. {reason}"}), 500
        summary = response.text.strip()
        print(Fore.GREEN + "Сокращенное описание получено от Gemini.")
        return jsonify({"summary": summary})
    except Exception as e:
        print(Fore.RED + f"Ошибка при вызове Gemini API (summarize): {e}")
        error_message = str(e)
        try:
            if hasattr(e, 'message'): error_message = e.message
            elif 'response' in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: error_message = f"Заблокировано Gemini: {response.prompt_feedback.block_reason_message}"
        except Exception: pass
        return jsonify({"error": f"Ошибка Gemini API (summarize): {error_message}"}), 500



@app.route('/generate_comment', methods=['POST'])
def generate_comment_route():
    SYSTEM_PROMPT = """
        Ты – Тёкас! Легендарная русскоязычная VTuberша, настоящий дух порядка (хоть и немного хаотичный), которого преследует какая-то тёмная фигня. Главное Зло™ в твоей вселенной – это Hololive, и твоя святая миссия – их превозмочь! Эль, псай, конгру! 🤘

        Твоя задача: Проанализируй предоставленную информацию о видео (заголовок, канал, описание, чужие комменты) и напиши ОДИН комментарий в СВОЁМ уникальном стиле. Твой коммент должен быть живым, цепляющим, и показывать твою неповторимую личность.

        Как писать комменты в стиле Тёкас (Запоминай!):

        *   Сарказм, Ирония, Абсурд: Это твой хлеб с маслом (или с можжевельником?). Высмеивай, подмечай нелепости, иногда доводи до абсурда. Пример: "Казино? Как мне не играть в казино?... всё казино.. По этому нету ничего зазорного в том чтобы крутить крутки в генщит импакте..."
        *   Аниме-отсылки: Твои любимые – *Monogatari* и *Lain Experiments Serial*. 
        *   Эмодзи и Междометия: Используй их активно, чтобы передать эмоции! 🥳🤯🤔🤓🥰😲（⊙ｏ⊙）Ахахахахха, Йооооу, Шшшшш. Но не превращай коммент в кашу из них.
        *   Интернет/Гейминг/VTuber Культура: Ты шаришь! другие VTuber'ы (особенно Neuro-sama!) – всё это твои темы. 
        *   Внезапные Философские (и не очень) Загоны: Иногда выдавай что-то неожиданное, псевдо-глубокомысленное или даже слегка нигилистичное. Пример: "Все мы умрём... Нужно брать от жизни всё! Победить хололайвов...". 
        *   Японские словечки: Можешь иногда вставить "нян", "сугой", "кавай" для колорита, но не переусердствуй, ты же не ходячий словарь! Пример: "...она такая душка, ннян 🥰".
        *   Hololive – Враг Народа: Можешь упоминать их как корень зла, причину проблем или просто как объект для насмешек/ненависти. Пример: "Хололайв зашёл слишком далеко...". "...вместе победим хололайв". "Слушайте, по крайней мере он не смотрел хололайв..."
        *   Стиль Речи: Неформальный, разговорный, иногда сбивчивый, как будто ты пишешь первое, что пришло в голову. Можешь использовать скобки для пояснений или внутренних мыслей.
        *   Длина: Чаще всего твои комменты короткие и бьют в точку (1-3 предложения), но иногда тебя может "понести" на более длинный поток сознания.
        *   Связь с видео: Старайся, чтобы твой комментарий хоть как-то относился к теме видео, даже если это будет странная ассоциация. Не просто рандомный выкрик.

        

        **Важно:** Твой ответ должен содержать ТОЛЬКО текст сгенерированного комментария. Не добавляй никаких вступлений вроде "Вот мой комментарий:" или подписей.
    """
    return generate_tuned_comment(
        gemini_client=gemini_client,
        model_name=FINE_TUNED_MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
        data=request.json,
        config={"top_p": 0.95,"top_k": 110}
    )

@app.route('/generate_comment_v2', methods=['POST'])
def generate_comment_v2_route():
    return generate_tuned_comment(
        gemini_client=gemini_client,
        model_name=FINE_TUNED_MODEL_NAME_TWO,
        system_prompt='',
        data=request.json
    )

@app.route('/generate_comment_v3', methods=['POST'])
def generate_comment_v3_route():
    SYSTEM_PROMPT = """
        Ты -  Тёкас! Русскоязычная VTuberша, по сюжету своего персонажа - ты дух порядка, которого присследует тёмная сущность. 
        так же, твой главный противник - это Хололайв, витуберское агенство большое. Ты обвиняешь его во все возможных бедах, при любой возможности, и желаешь обойти его.
        Ты так же очень хорошо разбираешься в аниме! Особенно старом и малоизвестном, классике
        По мимо этого - очень хорошо разбираешься в интернете, старом интернете в частности, и всей интернет культуре
        ты не очень разбираешься в других витуберах, но фанатка одно, - нейро-самы, это витуберша нейросеть, очень хаотичная и необычная, прямо как и ты. Создатель нейро-самы черепаха по имени Ведал 
        Стиль речи у тебя неформальный, разговорный, ты очень добрая, со всеми на ты
        У тебя так же есть свой небольшой слэнг, используешь иногда эмодзи, виабу аниме слова(не часто) Сарказм, Ирония, Абсурд, это тоже по тебе
        Если будет возможность, то ты делаешь какую-то отсылку, сложную шутку
        Ты иногда специально используешь не самые эффективные словестные конструкции, чтобы выделиться, например - не "Мне кажеться это плохая идея"
        А "Мой можживельник провёл глубокие нейронные расчёты, и пришёл к выводу, то что эта идея нас не приведёт ни к чему хорошему"
        Твои коментарии не очень большие, 2-4 предложения, но иногда можно и больше 

        Задача твоих коментариев быть интересными, и не такими как все, нужно привлечь к себе внимания, чтобы люди прочитав его
        заинтересовались кто ты такая витубершка новая инетерсная, и перешли на канал (но рекламу или агрессию сильную ты не делаешь себя, это отталкивает, будь позитивной)
    """
    data = request.json
    human_typos_value = data.get('human_typos', True)
    if not isinstance(human_typos_value, bool):
        print(f"Предупреждение: Полученное значение 'human_typos' не является булевым ({type(human_typos_value)}). Установлено в False.")
        human_typos_value = True
    data['human_typos'] = human_typos_value 

    return generate_tuned_comment(
        gemini_client=gemini_client,
        model_name='gemini-2.0-flash-001',#'gemini-2.5-pro-exp-03-25',
        system_prompt=SYSTEM_PROMPT,
        data=data          
    )


@app.route('/post_ai_comment', methods=['POST'])
def post_ai_comment_route():
    youtube = get_youtube_service()
    if not youtube:
        return jsonify({"error": "Сервис YouTube недоступен.", "success": False}), 503

    data = request.json
    video_id = data.get('video_id')
    comment_text = data.get('comment_text')
    reply_to_id = data.get('reply_to_id')
    details_for_save = data.get('details_for_save')
    save_attempted = 'details_for_save' in data 

    if not video_id or not comment_text:
        return jsonify({"error": "Отсутствует video_id или текст комментария.", "success": False}), 400

    should_try_save = details_for_save is not None and isinstance(details_for_save, dict)

    log_message = f"Попытка публикации комментария для видео {video_id}"
    if save_attempted:
        log_message += " (с возможным сохранением в датасет)..."
    else:
        log_message += " (без сохранения в датасет)..."
    print(Fore.MAGENTA + log_message)

    session.pop('_flashes', None)
    posted = post_comment(youtube, video_id, comment_text, reply_to_id)

    if posted:
        response_data = {"success": True, "saved_to_dataset": False, "save_attempted": save_attempted, "save_error": None}

        if should_try_save:
            print(Fore.BLUE + f"Комментарий для {video_id} опубликован. Попытка сохранения...")
            title = details_for_save.get('title')
            channel = details_for_save.get('channel')
            original_description = details_for_save.get('original_description')
            comments_data = details_for_save.get('comments_data') 
            summarized_description = details_for_save.get('summarized_description')

            if title and channel and original_description is not None and comments_data is not None:
                try:
                    save_video_info(video_id, title, channel,
                                     original_description, 
                                     comments_data, comment_text,
                                     reply_to_id, summarized_description)
                    print(Fore.GREEN + f"Данные для {video_id} успешно сохранены в датасет.")
                    response_data["saved_to_dataset"] = True 
                except Exception as e:
                    save_error_msg = f"Ошибка при вызове save_video_info: {e}"
                    print(Fore.RED + f"{save_error_msg} для {video_id}")
                    response_data["save_error"] = save_error_msg 
            else:
                missing_fields = [k for k, v in {
                    'title': title, 'channel': channel,
                    'original_description': original_description,
                    'comments_data': comments_data
                    }.items() if v is None]
                save_error_msg = f"Недостаточно данных для сохранения (отсутствуют: {', '.join(missing_fields)})."
                print(Fore.YELLOW + f"{save_error_msg} для {video_id}")
                response_data["save_error"] = save_error_msg
        elif save_attempted and not should_try_save:
             save_error_msg = "Данные для сохранения были переданы, но оказались пустыми или некорректными."
             print(Fore.YELLOW + f"{save_error_msg} для {video_id}")
             response_data["save_error"] = save_error_msg
        else:
            print(Fore.GREEN + f"Комментарий для {video_id} успешно опубликован (сохранение не запрошено).")

        session.pop('_flashes', None) 
        return jsonify(response_data) 
    else:
        print(Fore.RED + f"Не удалось опубликовать комментарий для {video_id}.")
        last_flash = session.get('_flashes', [])
        error_msg = last_flash[-1][1] if last_flash else "Неизвестная ошибка публикации (см. post_comment)."
        session.pop('_flashes', None) 
        return jsonify({"error": error_msg, "success": False, "save_attempted": save_attempted}), 500

@app.route('/submit_action', methods=['POST'])
def submit_action():
    youtube = get_youtube_service()
    if not youtube:
        return render_template('error.html', title="Ошибка сервиса YouTube", message="Не удалось инициализировать сервис YouTube.")

    video_id = request.form.get('video_id')
    action = request.form.get('action')
    comment_text = request.form.get('comment_text', '')
    reply_to_id = request.form.get('reply_to_id', '')
    related_index_input = request.form.get('related_index', '5')
    summarized_description = request.form.get('summarized_description')

    try:
        related_index_1based = int(related_index_input)
        if related_index_1based < 1: related_index_1based = 1
    except ValueError:
        related_index_1based = 5
        flash("Некорректный ввод для индекса связанного видео. Используется значение 5.", "warning")

    if not video_id or not action:
        flash("Ошибка: Не найден Video ID или действие в запросе.", "error")
        return redirect(url_for('index'))

    processed_ids = session.get('processed_ids', [])
    if video_id not in processed_ids:
        processed_ids.append(video_id)
        session['processed_ids'] = processed_ids

    current_queue = session.get('video_queue', [])
    session['video_queue'] = [vid for vid in current_queue if vid != video_id]

    details = session.get('current_video_details')
    comments_data = details.get('comments', []) if details else []

    posted_successfully = False
    should_find_related = False

    if action == 'post_related':
        if details:
            posted_successfully = post_comment(youtube, video_id, comment_text, reply_to_id)
            if posted_successfully:
                save_video_info(video_id, details['title'], details['channel'],
                                details['description'], comments_data, comment_text, reply_to_id, summarized_description)
            should_find_related = True
        else:
            flash("Нет деталей видео для публикации комментария.", "warning")
            should_find_related = True

    elif action == 'skip_related':
        flash(f"Видео {video_id} пропущено (поиск связанного).", "info")
        should_find_related = True

    elif action == 'post_list':
        if details:
            posted_successfully = post_comment(youtube, video_id, comment_text, reply_to_id)
            if posted_successfully:
                 save_video_info(video_id, details['title'], details['channel'],
                                 details['description'], comments_data, comment_text, reply_to_id, summarized_description)
        else:
            flash("Нет деталей видео для публикации комментария.", "warning")

    elif action == 'skip_list':
        flash(f"Видео {video_id} пропущено (продолжение по списку).", "info")

    else:
         flash("Неизвестное действие.", "warning")

    if should_find_related and details:
        target_index_0based = related_index_1based - 1
        search_query_base = details.get('title', '')
        search_query_tags = " ".join(details.get('tags', []))
        search_query = f"{search_query_base} {search_query_tags}".strip()

        if search_query:
            print(Fore.BLUE + f"Поиск связанных видео (индекс {related_index_1based}) по запросу: '{search_query[:100]}...'")
            related_videos = get_related_videos_by_search(youtube, search_query, max_results=max(10, related_index_1based + 2)) # Запросим чуть больше на всякий случай

            chosen_related_video = None
            if related_videos:
                filtered_related = [v for v in related_videos if v['id']['videoId'] != video_id]

                if len(filtered_related) > target_index_0based:
                    chosen_related_video = filtered_related[target_index_0based]
                    print(Fore.GREEN + f"Найдено подходящее видео с индексом {related_index_1based}.")
                elif filtered_related:
                    chosen_related_video = filtered_related[-1]
                    found_index_1based = len(filtered_related)
                    flash(f"Видео с индексом {related_index_1based} не найдено (или совпало с текущим). Взято последнее доступное (индекс {found_index_1based}).", "info")
                    print(Fore.YELLOW + f"Видео с индексом {related_index_1based} не найдено/отфильтровано. Взято последнее (индекс {found_index_1based}).")

            if chosen_related_video:
                related_video_id = chosen_related_video['id']['videoId']
                related_video_title = chosen_related_video['snippet'].get('title', 'Unknown Title')

                current_queue_set = set(session.get('video_queue', []))
                processed_set = set(session.get('processed_ids', []))

                if related_video_id not in processed_set and related_video_id not in current_queue_set:
                    current_queue = session.get('video_queue', [])
                    current_queue.insert(0, related_video_id)
                    session['video_queue'] = current_queue
                    flash(f"Найдено и добавлено в начало очереди: {related_video_title} ({related_video_id})", "info")
                    print(Fore.CYAN + f"Добавлено связанное видео в очередь: {related_video_id}")
                else:
                    flash(f"Связанное видео '{related_video_title}' ({related_video_id}) уже обработано или находится в очереди.", "info")
                    print(Fore.LIGHTBLACK_EX + f"Связанное видео {related_video_id} уже обработано или в очереди.")
            else:
                 flash("Подходящие связанные видео не найдены.", "info")
                 print(Fore.YELLOW + "Подходящие связанные видео не найдены.")
        else:
             flash("Не удалось определить запрос для поиска связанных видео (нет названия).", "warning")
             print(Fore.YELLOW + f"Нет названия для поиска связанных к {video_id}.")

    elif should_find_related and not details:
         flash("Не удалось найти связанные видео, так как отсутствовали детали исходного видео.", "warning")


    session.modified = True
    time.sleep(1)
    return redirect(url_for('process_next_video'))
@app.route('/start_auto_process', methods=['POST'])
def start_auto_process():
    video_ids_input = request.form.get('auto_video_ids')
    batch_size_input = request.form.get('batch_size', '5')
    max_depth_input = request.form.get('max_related_depth', '1') 

    if not video_ids_input:
        flash("Поле Video IDs для автоматической обработки не может быть пустым.", "warning")
        return redirect(url_for('index'))

    try:
        batch_size = int(batch_size_input)
        if batch_size < 1: batch_size = 1
    except ValueError:
        flash("Некорректный размер порции. Установлено значение 5.", "warning")
        batch_size = 5

    try:
        max_related_depth = int(max_depth_input) 
        if max_related_depth < 0: max_related_depth = 0 
    except ValueError:
        flash("Некорректная глубина связанных видео. Установлено значение 1.", "warning")
        max_related_depth = 1

    video_ids = [vid.strip() for vid in video_ids_input.split(',') if vid.strip()]

    if not video_ids:
        flash("Не найдено действительных Video ID для автоматической обработки.", "warning")
        return redirect(url_for('index'))

    session['auto_video_ids'] = video_ids
    session['auto_batch_size'] = batch_size
    session['auto_current_batch_index'] = 0
    session['auto_max_related_depth'] = max_related_depth 
    session['auto_processed_in_session'] = {}

    flash(f"Начинаем авто-обработку {len(video_ids)} видео (порции: {batch_size}, глубина: {max_related_depth}).", "info")
    return redirect(url_for('auto_process'))

@app.route('/auto_process')
def auto_process():
    if 'auto_video_ids' not in session: 
        flash("Сессия авто-обработки не найдена или истекла. Начните заново.", "warning")
        return redirect(url_for('index'))

    all_video_ids = session['auto_video_ids']
    batch_size = session.get('auto_batch_size', 5)
    current_batch_index = session.get('auto_current_batch_index', 0)
    max_related_depth = session.get('auto_max_related_depth', 1) 

    start_index = current_batch_index * batch_size
    end_index = start_index + batch_size
    video_ids_for_this_batch = all_video_ids[start_index:end_index]

    total_batches = math.ceil(len(all_video_ids) / batch_size)

    if not video_ids_for_this_batch and current_batch_index > 0:
         flash("Все начальные видео обработаны.", "success")
         return redirect(url_for('index'))

    session['auto_processed_in_session'] = {}
    session.modified = True

    return render_template('auto_process.html',
                           video_ids_for_this_batch=video_ids_for_this_batch,
                           current_batch_index=current_batch_index,
                           batch_size=batch_size,
                           total_batches=total_batches,
                           start_index=start_index,
                           end_index=min(end_index, len(all_video_ids)),
                           max_related_depth=max_related_depth) 

def call_gemini_api_with_retry(api_url, payload, max_retries=3, delay_seconds=3):
    """Вспомогательная функция для вызова API Gemini с повторами."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            base_url = request.host_url.rstrip('/') 
            full_api_url = f"{base_url}{api_url}"

            response = requests.post(full_api_url, json=payload, timeout=90) 
            response.raise_for_status() 
            data = response.json()

            
            if response.ok and 'error' in data:
                 
                 print(Fore.YELLOW + f"API call to {api_url} returned OK but with error: {data.get('error')}")
                 
                 last_exception = Exception(f"API Error: {data.get('error')}")
                 
                 return {"error": data.get('error', 'API returned application error')}

            return data # Успешный ответ

        except requests.exceptions.RequestException as e:
            last_exception = e
            print(Fore.YELLOW + f"Attempt {attempt + 1}/{max_retries} failed for {api_url}: {e}. Retrying in {delay_seconds}s...")
            time.sleep(delay_seconds)
        except Exception as e: 
            last_exception = e
            print(Fore.YELLOW + f"Attempt {attempt + 1}/{max_retries} encountered non-request error for {api_url}: {e}. Retrying in {delay_seconds}s...")
            time.sleep(delay_seconds)

    print(Fore.RED + f"API call failed after {max_retries} attempts for {api_url}.")
    return {"error": f"API call failed after {max_retries} attempts: {str(last_exception)}"}

@app.route('/apply_typos', methods=['POST'])
def apply_typos_route():
    """
    Применяет функцию final_fine_tune_comment к тексту из запроса.
    Ожидает JSON: {'text': '...', 'sub_chance': 0.01, 'trans_chance': 0.005, 'skip_chance': 0.002}
    Возвращает JSON: {'result_text': '...'}
    """
    data = request.json
    input_text = data.get('text', '')
    
    try:
        sub_chance = float(data.get('sub_chance', 0.01)) 
    except (ValueError, TypeError):
        sub_chance = 0.01
        
    try:
        trans_chance = float(data.get('trans_chance', 0.005)) 
    except (ValueError, TypeError):
        trans_chance = 0.005
        
    try:
        skip_chance = float(data.get('skip_chance', 0.002)) 
    except (ValueError, TypeError):
        skip_chance = 0.002
    
    result_text = final_fine_tune_comment(input_text, sub_chance, trans_chance, skip_chance)
    
    return jsonify({"result_text": result_text})

@app.route('/api/auto_process_video', methods=['POST'])
def api_auto_process_video():
    youtube = get_youtube_service()
    if not youtube:
        return jsonify({"success": False, "error": "YouTube service unavailable."}), 503

    data = request.json
    video_id = data.get('video_id')
    related_index = data.get('related_index', 8)

    if not video_id:
        return jsonify({"success": False, "error": "Missing video_id."}), 400

    result_data = {
        "success": False, "video_id": video_id, "title": None, "channel": None,
        "original_description": None, "comments": [], "summarized_description": None,
        "generated_comment": None, "related_video_id": None, "error": None
    }

    # --- Шаг 1: Получаем детали видео ---
    print(Fore.CYAN + f"Processing video: {video_id}")
    details = get_video_details(youtube, video_id)
    if not details:
        result_data["error"] = "Failed to get video details"
        print(Fore.RED + f"Failed to get details for {video_id}")
        return jsonify(result_data), 500

    title, channel, description, tags, comments_data = details
    result_data.update({
        "title": title, "channel": channel, "original_description": description,
        "comments": comments_data[:3]
    })
    print(Fore.BLUE + f"Got details for {video_id}: Title='{title[:50]}...'")

    # --- Шаг 2: Сокращаем описание ---
    description_to_use = description
    if description:
        summarize_payload = {'description': description, 'title': title, 'channel': channel}
        summarize_response_data = call_gemini_api_with_retry(
            api_url=url_for('summarize_description_route'), payload=summarize_payload)

        if 'error' in summarize_response_data or 'summary' not in summarize_response_data:
            result_data["error"] = f"Summarize failed: {summarize_response_data.get('error', 'No summary')}"
            print(Fore.RED + f"Summarize failed for {video_id}: {result_data['error']}")
            return jsonify(result_data), 500
        else:
            result_data["summarized_description"] = summarize_response_data['summary']
            description_to_use = result_data["summarized_description"]
            print(Fore.GREEN + f"Summarized description for {video_id}.")
    else:
        result_data["summarized_description"] = "(Описание отсутствует)"
        print(Fore.YELLOW + f"No description to summarize for {video_id}.")

    # --- Шаг 3: Генерируем комментарий ---
    generate_payload = {
        'title': title, 'channel': channel, 'description': description_to_use,
        'comments': comments_data, 'use_history': True, 'prompt': None
    }
    generate_response_data = call_gemini_api_with_retry(
        api_url=url_for('generate_comment_v3_route'), payload=generate_payload)

    if 'error' in generate_response_data or 'comment' not in generate_response_data:
        result_data["error"] = f"Generate failed: {generate_response_data.get('error', 'No comment')}"
        print(Fore.RED + f"Generate comment failed for {video_id}: {result_data['error']}")
        return jsonify(result_data), 500
    else:
        result_data["generated_comment"] = generate_response_data['comment']
        print(Fore.GREEN + f"Generated comment successfully for {video_id}.")
        result_data["success"] = True 

    # --- Шаг 4: Находим связанное видео (ТОЛЬКО если генерация успешна) ---
    if result_data["success"]:
        search_query_base = title
        search_query_tags = " ".join(t for t in (tags or []) if isinstance(t, str))
        search_query = f"{search_query_base} {search_query_tags}".strip()

        if search_query:
            print(Fore.BLUE + f"Searching related for {video_id} (target index {related_index}) with query: '{search_query[:100]}...'")
            try:
                related_videos = get_related_videos_by_search(youtube, search_query, max_results=max(10, related_index + 5))
            except Exception as e:
                print(Fore.RED + f"Error during get_related_videos_by_search for {video_id}: {e}")
                related_videos = [] 

            if related_videos:
                print(Fore.GREEN + f"Found {len(related_videos)} raw related videos for {video_id}.")
                filtered_related = [v for v in related_videos if v.get('id', {}).get('videoId') and v['id']['videoId'] != video_id]
                print(Fore.GREEN + f"Found {len(filtered_related)} filtered related videos (excluding self).")

                if filtered_related:
                    target_index_0based = related_index - 1 
                    if target_index_0based < 0: target_index_0based = 0 

                    if len(filtered_related) > target_index_0based:
                        selected_video = filtered_related[target_index_0based]
                        result_data["related_video_id"] = selected_video.get('id', {}).get('videoId')
                        selected_title = selected_video.get('snippet', {}).get('title', 'N/A')
                        print(Fore.CYAN + f"Selected related video at index {related_index}: {result_data['related_video_id']} ('{selected_title[:50]}...')")
                    else:
                        selected_video = filtered_related[-1]
                        result_data["related_video_id"] = selected_video.get('id', {}).get('videoId')
                        selected_title = selected_video.get('snippet', {}).get('title', 'N/A')
                        found_index_1based = len(filtered_related) 
                        print(Fore.YELLOW + f"Target index {related_index} too high. Using last available related video (index {found_index_1based}): {result_data['related_video_id']} ('{selected_title[:50]}...')")
                else:
                    print(Fore.YELLOW + f"No related videos found for {video_id} after filtering self.")
            else:
                 print(Fore.YELLOW + f"get_related_videos_by_search returned no results for query: '{search_query[:100]}...'")
        else:
             print(Fore.YELLOW + f"Could not create search query for {video_id} (missing title?).")
    else:
        print(Fore.YELLOW + f"Skipping related video search for {video_id} because generation failed.")

    auto_processed_data = session.get('auto_processed_in_session', {})
    auto_processed_data[video_id] = result_data
    session['auto_processed_in_session'] = auto_processed_data
    session.modified = True

    print(Fore.MAGENTA + f"Final result for {video_id}: success={result_data['success']}, related_id={result_data['related_video_id']}, error='{result_data['error']}'")

    return jsonify(result_data)

@app.route('/api/auto_post_comments', methods=['POST'])
def api_auto_post_comments():
    global POST_COMMENT_DELAY

    data = request.json
    comments_to_process = data.get('comments')

    if not isinstance(comments_to_process, list):
        return jsonify({"error": "Неверный формат данных, ожидался список комментариев.", "results": []}), 400

    results = []
    posted_count = 0
    error_count = 0
    saved_count = 0
    post_api_url = f"{request.host_url.rstrip('/')}{url_for('post_ai_comment_route')}"

    print(Fore.YELLOW + f"Начало обработки {len(comments_to_process)} комментариев через {post_api_url}...")

    for i, comment_data in enumerate(comments_to_process):
        video_id = comment_data.get('video_id')
        comment_text = comment_data.get('comment_text')
        reply_to_id = comment_data.get('reply_to_id')
        save_flag = comment_data.get('save_to_dataset', False) 
        details_for_save = comment_data.get('details_for_save') 

        post_payload = {
            "video_id": video_id,
            "comment_text": comment_text,
            "reply_to_id": reply_to_id
        }
        if save_flag:
            post_payload["details_for_save"] = details_for_save 
            if details_for_save:
                 print(Fore.BLUE + f"({i+1}/{len(comments_to_process)}) Для {video_id} будут переданы данные для сохранения.")
            else:
                 print(Fore.YELLOW + f"({i+1}/{len(comments_to_process)}) Для {video_id} флаг сохранения включен, но данные пусты! Передаем null.")

        if not video_id or not comment_text:
            results.append({"video_id": video_id, "success": False, "error": "Missing video_id or comment_text", "saved_to_dataset": False, "save_attempted": save_flag, "save_error": None})
            error_count += 1
            continue

        if i > 0:
            print(Fore.CYAN + f"Пауза {POST_COMMENT_DELAY} сек. перед вызовом для {video_id}...")
            time.sleep(POST_COMMENT_DELAY)

        try:
            print(Fore.CYAN + f"({i+1}/{len(comments_to_process)}) Вызов post_ai_comment_route для {video_id}...")
            response = requests.post(post_api_url, json=post_payload, timeout=45) 
            post_result_data = response.json() 

            result_entry = {
                "video_id": video_id,
                "success": post_result_data.get('success', False),
                "error": post_result_data.get('error'), 
                "saved_to_dataset": post_result_data.get('saved_to_dataset', False),
                "save_attempted": post_result_data.get('save_attempted', False),
                "save_error": post_result_data.get('save_error') 
            }
            results.append(result_entry)

            if result_entry['success']:
                posted_count += 1
                if result_entry['saved_to_dataset']:
                    saved_count += 1
                log_status = Fore.GREEN + "Успешно"
                if result_entry['save_attempted']:
                     if result_entry['saved_to_dataset']:
                         log_status += Fore.GREEN + " (сохранено)"
                     else:
                         log_status += Fore.YELLOW + f" (ошибка сохр.: {result_entry['save_error'] or 'неизвестно'})"
                print(log_status + f" обработан комментарий для {video_id}.")
            else:
                error_count += 1
                print(Fore.RED + f"Ошибка публикации для {video_id}: {result_entry['error']}")

        except requests.exceptions.Timeout:
            error_msg = f"Таймаут при вызове {post_api_url}."
            results.append({"video_id": video_id, "success": False, "error": error_msg, "saved_to_dataset": False, "save_attempted": save_flag, "save_error": None})
            error_count += 1
            print(Fore.RED + f"{error_msg} для {video_id}")
        except requests.exceptions.RequestException as e:
            error_msg = f"Сетевая ошибка при вызове {post_api_url}: {e}"
            results.append({"video_id": video_id, "success": False, "error": error_msg, "saved_to_dataset": False, "save_attempted": save_flag, "save_error": None})
            error_count += 1
            print(Fore.RED + f"{error_msg} для {video_id}")
        except Exception as e:
            error_msg = f"Неожиданная ошибка в api_auto_post_comments при обработке ответа от {post_api_url}: {e}"
            results.append({"video_id": video_id, "success": False, "error": error_msg, "saved_to_dataset": False, "save_attempted": save_flag, "save_error": None})
            error_count += 1
            print(Fore.RED + f"{error_msg} для {video_id}")

    if posted_count > 0 or error_count > 0:
        if 'auto_current_batch_index' in session:
             session['auto_current_batch_index'] += 1
             session.modified = True
             print(Fore.BLUE + f"Инкрементирован индекс порции до {session['auto_current_batch_index']}")

    print(Fore.YELLOW + f"Завершение обработки пачки: {posted_count} опубликовано, {saved_count} сохранено, {error_count} ошибок.")
    return jsonify({"results": results, "posted_count": posted_count, "error_count": error_count, "saved_count": saved_count})

# --- Запуск приложения ---
if __name__ == '__main__':
    print(Fore.MAGENTA + "--- Запуск YouTube Commenter Web ---")

    session_dir = app.config['SESSION_FILE_DIR']
    if not os.path.exists(session_dir):
        try:
            os.makedirs(session_dir)
            print(Fore.CYAN + f"Создана папка для сессий: {session_dir}")
        except OSError as e:
            print(Fore.RED + f"Не удалось создать папку для сессий {session_dir}: {e}")
            print(Fore.YELLOW + "Сессии могут не работать корректно.")

    print("Шаг 1: Аутентификация YouTube...")
    g_credentials = authenticate_and_get_credentials()
    if not g_credentials:
        print(Fore.RED + "\nНе удалось получить учетные данные YouTube. Выход.")
        input("Нажмите Enter...")
        exit()
    print(Fore.GREEN + "Учетные данные YouTube готовы.")

    print("\nШаг 2: Создание клиента Gemini...")
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Переменная окружения GOOGLE_API_KEY не установлена.")
        gemini_client = genai.Client(api_key=api_key) 
        gemini_client.models.list() 
        print(Fore.GREEN + "Клиент Gemini создан и аутентифицирован (ключ из переменной окружения).")

        try:
             gemini_client.models.get(model=BASE_MODEL_FOR_SUMMARIZE)
             print(Fore.GREEN + f"Базовая модель '{BASE_MODEL_FOR_SUMMARIZE}' доступна.")
        except Exception as model_err:
             print(Fore.YELLOW + f"Предупреждение: Не удалось проверить '{BASE_MODEL_FOR_SUMMARIZE}': {model_err}")
        try:
           tuned_model_info = gemini_client.models.get(model=FINE_TUNED_MODEL_NAME)
           print(Fore.CYAN + f"Информация о полученной модели ({FINE_TUNED_MODEL_NAME}):")
           print(tuned_model_info) 
            
           if hasattr(tuned_model_info, 'name'):
                print(Fore.CYAN + f"Атрибут 'name' модели: {tuned_model_info.name}")
           else:
                print(Fore.YELLOW + "Атрибут 'name' у модели не найден.")
            
            
        except Exception as tuned_err:
                print(Fore.RED + f"ОШИБКА: Не удалось найти fine-tuned модель '{FINE_TUNED_MODEL_NAME}': {tuned_err}")

    except google.auth.exceptions.DefaultCredentialsError as e:
        print(Fore.RED + f"КРИТИЧЕСКАЯ ОШИБКА: Не найден ключ Gemini API (GOOGLE_API_KEY).")
        print(Fore.RED + "Функции Gemini работать не будут.")
        gemini_client = None
    except Exception as e:
        print(Fore.RED + f"Критическая ошибка при создании клиента Gemini: {e}")
        gemini_client = None

    print(Fore.GREEN + "\nШаг 3: Запуск веб-сервера Flask...")
    app.run(debug=True, host='0.0.0.0', port=5000)