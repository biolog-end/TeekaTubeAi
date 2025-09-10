import os
import json

# --- Настройки ---
OLD_TXT_FILE = "video_info.txt"  # Имя твоего старого TXT файла
DATASET_JSON_FILE = "training_dataset.json"   # Имя твоего JSON файла с датасетом

def parse_old_txt_to_json(txt_filepath, json_filepath):
    """
    Парсит старый TXT файл и добавляет данные в JSON датасет.
    """
    new_data_pairs = []
    processed_count = 0
    error_count = 0

    print(f"Начинаю обработку файла: {txt_filepath}")

    if not os.path.exists(txt_filepath):
        print(f"Ошибка: Файл '{txt_filepath}' не найден.")
        return

    try:
        with open(txt_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Ошибка чтения файла '{txt_filepath}': {e}")
        return

    # Разделяем файл на блоки по разделителю '==='
    video_blocks = content.strip().split('===\n') # Используем '\n' чтобы не захватывать '===' в конце

    print(f"Найдено {len(video_blocks)} блоков для обработки.")

    for i, block in enumerate(video_blocks):
        block = block.strip()
        if not block:
            continue

        print(f"\n--- Обработка блока {i+1} ---")
        lines = block.split('\n')

        video_info = {
            "title": None,
            "channel": None,
            "description": [], # Используем список для многострочного описания
            "msg1": None,
            "msg2": None,
            "msg3": None,
            "comment": None
        }
        
        is_reading_description = False

        try:
            current_line_index = 0
            while current_line_index < len(lines):
                line = lines[current_line_index].strip()

                if line.startswith("Video name:"):
                    video_info["title"] = line.replace("Video name:", "", 1).strip()
                    is_reading_description = False
                elif line.startswith("Channel name:"):
                    video_info["channel"] = line.replace("Channel name:", "", 1).strip()
                    is_reading_description = False
                elif line.startswith("Description:"):
                    # Начинаем читать описание, захватываем первую строку
                    first_desc_line = line.replace("Description:", "", 1).strip()
                    if first_desc_line: # Добавляем только если не пустая
                        video_info["description"].append(first_desc_line)
                    is_reading_description = True
                elif line.startswith("Message 1:"):
                    video_info["msg1"] = line.replace("Message 1:", "", 1).strip()
                    is_reading_description = False
                elif line.startswith("Message 2:"):
                    video_info["msg2"] = line.replace("Message 2:", "", 1).strip()
                    is_reading_description = False
                elif line.startswith("Message 3:"):
                    video_info["msg3"] = line.replace("Message 3:", "", 1).strip()
                    is_reading_description = False
                elif line.startswith("Comment:"):
                    video_info["comment"] = line.replace("Comment:", "", 1).strip()
                    is_reading_description = False
                elif is_reading_description:
                    # Если читаем описание и строка не начинается с известного префикса,
                    # добавляем ее к описанию
                    video_info["description"].append(line)
                
                current_line_index += 1
            
            # --- Формируем пару для JSON ---
            if video_info["title"] and video_info["channel"] and video_info["comment"]:
                # Собираем описание из списка строк
                full_description = "\n".join(video_info["description"]).strip()

                # Формируем входную строку (контекст)
                input_parts = [
                    f"Title: {video_info['title']}",
                    f"Channel: {video_info['channel']}",
                    f"Description:\n{full_description if full_description else 'Нет описания'}",
                    "--- Comments ---",
                    f"Comment 1: {video_info['msg1'] if video_info['msg1'] else 'N/A'}",
                    f"Comment 2: {video_info['msg2'] if video_info['msg2'] else 'N/A'}",
                    f"Comment 3: {video_info['msg3'] if video_info['msg3'] else 'N/A'}"
                ]
                input_string = "\n".join(input_parts)

                # Выходная строка - это комментарий пользователя
                output_string = video_info["comment"]

                new_data_pairs.append([input_string, output_string])
                print(f"Блок {i+1}: Успешно обработан.")
                processed_count += 1
            else:
                print(f"Блок {i+1}: Пропущен - не найдены обязательные поля (Title, Channel, Comment).")
                error_count += 1

        except Exception as e:
            print(f"Блок {i+1}: Ошибка при обработке блока: {e}")
            error_count += 1

    print(f"\nОбработка TXT завершена. Успешно: {processed_count}, Ошибки/Пропущено: {error_count}.")

    if not new_data_pairs:
        print("Нет новых данных для добавления в JSON.")
        return

    # --- Загрузка и обновление JSON файла ---
    dataset = []
    if os.path.exists(json_filepath):
        print(f"Загрузка существующего датасета из: {json_filepath}")
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "dataset" in data and isinstance(data["dataset"], list):
                    dataset = data["dataset"]
                    print(f"Загружено {len(dataset)} существующих записей.")
                else:
                    print(f"Предупреждение: Файл {json_filepath} имеет неверную структуру. Начинаем с пустого списка.")
        except json.JSONDecodeError:
            print(f"Предупреждение: Не удалось декодировать JSON из {json_filepath}. Начинаем с пустого списка.")
        except Exception as e:
            print(f"Предупреждение: Не удалось прочитать {json_filepath}: {e}. Начинаем с пустого списка.")

    # Добавляем новые данные к существующим
    original_count = len(dataset)
    dataset.extend(new_data_pairs)
    added_count = len(dataset) - original_count

    print(f"Добавлено {added_count} новых записей. Всего записей: {len(dataset)}.")

    # Сохраняем обновленный датасет
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump({"dataset": dataset}, f, indent=4, ensure_ascii=False)
        print(f"Обновленный датасет успешно сохранен в: {json_filepath}")
    except Exception as e:
        print(f"Ошибка сохранения датасета в JSON файл '{json_filepath}': {e}")

# --- Запуск скрипта ---
if __name__ == "__main__":
    parse_old_txt_to_json(OLD_TXT_FILE, DATASET_JSON_FILE)
    print("\nСкрипт завершил работу.")