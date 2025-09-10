# Сначала нужно установить библиотеку для работы с эмодзи:
# pip install emoji
import emoji

def final_fine_tune_comment(comment: str) -> str:
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

    return cleaned_comment

# --- Примеры использования ---

# Пример 1: Замена " и удаление точки
comment1 = "Это тестовый комментарий &quot;с цитатой&quot;."
print(f"Original: '{comment1}'")
print(f"Cleaned:  '{final_fine_tune_comment(comment1)}'")
# Ожидаемый результат: 'Это тестовый комментарий "с цитатой"'

# Пример 2: Удаление точки (как в твоем примере)
comment2 = "Матвей! Монагатари - крутое аниме, советую, если не смотрели."
print(f"Original: '{comment2}'")
print(f"Cleaned:  '{final_fine_tune_comment(comment2)}'")
# Ожидаемый результат: 'Матвей! Монагатари - крутое аниме, советую, если не смотрели'

# Пример 3: Многоточие не трогаем
comment3 = "Извини, Максим, но это твой пердущий мячик..."
print(f"Original: '{comment3}'")
print(f"Cleaned:  '{final_fine_tune_comment(comment3)}'")
# Ожидаемый результат: 'Извини, Максим, но это твой пердущий мячик...'

# Пример 4: Точка перед эмодзи
comment4 = "Так вот как они штампуют это. 🤖🤬"
print(f"Original: '{comment4}'")
print(f"Cleaned:  '{final_fine_tune_comment(comment4)}'")
# Ожидаемый результат: 'Так вот как они штампуют это 🤖🤬'

# Пример 5: Многоточие перед эмодзи
comment5 = "Так вот как они штампуют это... 😋🥰"
print(f"Original: '{comment5}'")
print(f"Cleaned:  '{final_fine_tune_comment(comment5)}'")
# Ожидаемый результат: 'Так вот как они штампуют это... 😋🥰'

# Пример 6: Точка перед пробелом
comment6 = "Просто текст. "
print(f"Original: '{comment6}'")
print(f"Cleaned:  '{final_fine_tune_comment(comment6)}'")
# Ожидаемый результат: 'Просто текст '

# Пример 7: Многоточие перед пробелом
comment7 = "Просто текст... "
print(f"Original: '{comment7}'")
print(f"Cleaned:  '{final_fine_tune_comment(comment7)}'")
# Ожидаемый результат: 'Просто текст... '

# Пример 8: Только точка
comment8 = "."
print(f"Original: '{comment8}'")
print(f"Cleaned:  '{final_fine_tune_comment(comment8)}'")
# Ожидаемый результат: ''

# Пример 9: Только многоточие
comment9 = "..."
print(f"Original: '{comment9}'")
print(f"Cleaned:  '{final_fine_tune_comment(comment9)}'")
# Ожидаемый результат: '...'