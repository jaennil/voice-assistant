#!/usr/bin/env python3
"""
Голосовой ассистент с wake word detection.
Постоянно слушает микрофон и реагирует на ключевое слово "санёк".
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
WAKE_WORD = "санек"  # vosk распознаёт без ё
WAKE_WORD_VARIANTS = ["санек", "саня", "санёк"]  # варианты написания
MODEL_PATH = "/home/jaennil/.local/share/vosk/vosk-model-small-ru-0.22"
SAMPLE_RATE = 16000
SILENCE_TIMEOUT = 2.0  # секунды тишины для окончания диктовки

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Очередь для аудио данных
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
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


def contains_wake_word(text: str) -> bool:
    """Проверяет наличие wake word в тексте."""
    text_lower = text.lower()
    return any(word in text_lower for word in WAKE_WORD_VARIANTS)


def extract_after_wake_word(text: str) -> str:
    """Извлекает текст после wake word."""
    text_lower = text.lower()
    for word in WAKE_WORD_VARIANTS:
        if word in text_lower:
            idx = text_lower.find(word)
            return text[idx + len(word):].strip()
    return ""


def listen_for_dictation(recognizer: KaldiRecognizer) -> str:
    """Слушает диктовку до 2 секунд тишины и возвращает весь текст."""
    log.info("🎤 Слушаю диктовку (2 сек тишины для завершения)...")
    text_parts = []
    last_speech_time = time.time()

    while True:
        try:
            data = audio_queue.get(timeout=0.1)
        except queue.Empty:
            # Проверяем таймаут тишины
            if time.time() - last_speech_time >= SILENCE_TIMEOUT:
                log.info("⏹️ 2 секунды тишины - завершаю диктовку")
                break
            continue

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                log.info(f"Распознано: '{text}'")
                text_parts.append(text)
                last_speech_time = time.time()
        else:
            # Частичный результат - сбрасываем таймер тишины
            partial = json.loads(recognizer.PartialResult())
            if partial.get("partial", "").strip():
                last_speech_time = time.time()

        # Проверяем таймаут тишины
        if time.time() - last_speech_time >= SILENCE_TIMEOUT:
            log.info("⏹️ 2 секунды тишины - завершаю диктовку")
            break

    return " ".join(text_parts)


def main():
    log.info("Загрузка модели...")
    try:
        model = Model(MODEL_PATH)
    except Exception as e:
        log.error(f"Не удалось загрузить модель: {e}")
        sys.exit(1)

    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)

    log.info(f"👂 Жду wake word: '{WAKE_WORD}'")
    log.info("Варианты: " + ", ".join(WAKE_WORD_VARIANTS))

    # Используем pipewire для автоматического ресемплинга
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

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if text:
                        log.info(f"Услышал: '{text}'")

                    if contains_wake_word(text):
                        log.info("✨ Wake word обнаружен!")

                        # Собираем весь текст
                        all_text_parts = []

                        # Проверяем, есть ли текст после wake word
                        remainder = extract_after_wake_word(text)
                        if remainder:
                            all_text_parts.append(remainder)

                        # Слушаем дальнейшую диктовку
                        dictation = listen_for_dictation(recognizer)
                        if dictation:
                            all_text_parts.append(dictation)

                        # Печатаем всё сразу
                        full_text = " ".join(all_text_parts)
                        if full_text:
                            type_text(full_text)

                        log.info(f"👂 Жду wake word: '{WAKE_WORD}'")

    except KeyboardInterrupt:
        log.info("Завершение работы...")
    except Exception as e:
        log.error(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
