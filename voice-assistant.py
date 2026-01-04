#!/usr/bin/env python3
"""
Голосовой ассистент с wake word detection.
Поддержка русского и английского языков с автоопределением.
Wake words: "компьютер" (RU), "computer" (EN)
"""

import subprocess
import json
import queue
import logging
import sys
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Настройки
SAMPLE_RATE = 16000
SILENCE_TIMEOUT = 2.0  # секунды тишины для окончания диктовки

# Языковые настройки
LANGUAGES = {
    "ru": {
        "model_path": "/home/jaennil/.local/share/vosk/vosk-model-small-ru-0.22",
        "wake_words": ["компьютер", "компютер"],
        "name": "Русский"
    },
    "en": {
        "model_path": "/home/jaennil/.local/share/vosk/vosk-model-small-en-us-0.15",
        "wake_words": ["computer"],
        "name": "English"
    }
}

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Очередь для аудио данных
audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """Callback для захвата аудио."""
    if status:
        log.warning(f"Audio status: {status}")
    audio_queue.put(bytes(indata))


def type_text(text: str):
    """Печатает текст в активное окно через xdotool."""
    if not text.strip():
        return
    log.info(f"Печатаю: {text}")
    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--", text],
            check=True,
            timeout=10
        )
    except subprocess.SubprocessError as e:
        log.error(f"Ошибка xdotool: {e}")


def check_wake_word(text: str, lang: str) -> tuple[bool, str]:
    """Проверяет наличие wake word и возвращает (найден, остаток текста)."""
    text_lower = text.lower()
    for word in LANGUAGES[lang]["wake_words"]:
        if word in text_lower:
            idx = text_lower.find(word)
            remainder = text[idx + len(word):].strip()
            return True, remainder
    return False, ""


def listen_for_dictation(recognizer: KaldiRecognizer, lang: str) -> str:
    """Слушает диктовку до 2 секунд тишины и возвращает весь текст."""
    lang_name = LANGUAGES[lang]["name"]
    log.info(f"🎤 Слушаю диктовку [{lang_name}] (2 сек тишины для завершения)...")
    text_parts = []
    last_speech_time = time.time()

    while True:
        try:
            data = audio_queue.get(timeout=0.1)
        except queue.Empty:
            if time.time() - last_speech_time >= SILENCE_TIMEOUT:
                log.info("⏹️ 2 секунды тишины - завершаю диктовку")
                break
            continue

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                log.info(f"Распознано [{lang_name}]: '{text}'")
                text_parts.append(text)
                last_speech_time = time.time()
        else:
            partial = json.loads(recognizer.PartialResult())
            if partial.get("partial", "").strip():
                last_speech_time = time.time()

        if time.time() - last_speech_time >= SILENCE_TIMEOUT:
            log.info("⏹️ 2 секунды тишины - завершаю диктовку")
            break

    return " ".join(text_parts)


def main():
    # Загрузка моделей
    models = {}
    recognizers = {}

    for lang, config in LANGUAGES.items():
        log.info(f"Загрузка модели: {config['name']}...")
        try:
            models[lang] = Model(config["model_path"])
            recognizers[lang] = KaldiRecognizer(models[lang], SAMPLE_RATE)
            recognizers[lang].SetWords(True)
            log.info(f"  Wake words: {', '.join(config['wake_words'])}")
        except Exception as e:
            log.error(f"Не удалось загрузить модель {config['name']}: {e}")
            sys.exit(1)

    log.info("👂 Жду wake word: 'компьютер' (RU) или 'computer' (EN)")

    device = "pipewire"
    log.info(f"Используем аудио устройство: {device}")

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=4000,
            dtype='int16',
            channels=1,
            device=device,
            callback=audio_callback
        ):
            while True:
                try:
                    data = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Проверяем оба языка
                detected_lang = None
                remainder = ""

                for lang, rec in recognizers.items():
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "").strip()

                        if text:
                            log.info(f"Услышал [{LANGUAGES[lang]['name']}]: '{text}'")

                        found, rem = check_wake_word(text, lang)
                        if found:
                            detected_lang = lang
                            remainder = rem
                            break

                if detected_lang:
                    lang_name = LANGUAGES[detected_lang]["name"]
                    log.info(f"✨ Wake word обнаружен! Язык: {lang_name}")

                    # Создаём свежий распознаватель для диктовки
                    dict_recognizer = KaldiRecognizer(models[detected_lang], SAMPLE_RATE)
                    dict_recognizer.SetWords(True)

                    all_text_parts = []
                    if remainder:
                        all_text_parts.append(remainder)

                    dictation = listen_for_dictation(dict_recognizer, detected_lang)
                    if dictation:
                        all_text_parts.append(dictation)

                    full_text = " ".join(all_text_parts)
                    if full_text:
                        type_text(full_text)

                    # Сбрасываем распознаватели
                    for rec in recognizers.values():
                        rec.Reset()

                    log.info("👂 Жду wake word: 'компьютер' (RU) или 'computer' (EN)")

    except KeyboardInterrupt:
        log.info("Завершение работы...")
    except Exception as e:
        log.error(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
