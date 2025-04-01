import pyaudio
import io
import wave
from libs.play_sounds import pyaudio_play_audio_function

sample_rate = 16000 # частота дискретизации должна 
                    # совпадать при синтезе и воспроизведении


def record_audio(seconds, sample_rate, 
                 chunk_size=4000, num_channels=1) -> bytes:
    """
    Записывает аудио данной продолжительности и возвращает бинарный объект с данными
    
    :param integer seconds: Время записи в секундах
    :param integer sample_rate: частота дискретизации, такая же 
        какую вы указали в параметре sampleRateHertz
    :param integer chunk_size: размер семпла записи
    :param integer num_channels: количество каналов, в режимер синхронного
        распознавания спичкит принимает моно дорожку, 
        поэтому стоит оставить значение `1`
    :return: Возвращает объект BytesIO с аудио данными в формате WAV
    :rtype: bytes
    """

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=num_channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )
    frames = []
    try:
        for i in range(0, int(sample_rate / chunk_size * seconds)):
            data = stream.read(chunk_size)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    container = io.BytesIO()
    wf = wave.open(container, 'wb')
    wf.setnchannels(num_channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    container.seek(0)
    return container

from speechkit import Session, SpeechSynthesis
from speechkit import ShortAudioRecognition
#oauth_token = "y0_AgAEA7qh4_wyAATuwQAAAAD5mqOdAACu7MMZKnZGorfs4oNGiNiEjY9I0g"
catalog_id = "b1g3mrejv8qqnnligebe"
oauth_token = "y0__xCy-I-PqveAAhjB3RMg9I_ZoxKOAiI_mc80U6mYhWF1nsmKEpeJug"


# Экземпляр класса `Session` можно получать из разных данных 
session = Session.from_yandex_passport_oauth_token(oauth_token, catalog_id)

# Создаем экземляр класса `SpeechSynthesis`, передавая `session`,
# который уже содержит нужный нам IAM-токен 
# и другие необходимые для API реквизиты для входа
synthesizeAudio = SpeechSynthesis(session)

# Создаем экземпляр класса с помощью `session` полученного ранее
recognizeShortAudio = ShortAudioRecognition(session)

def generate_speech(phraze):
    audio_data = synthesizeAudio.synthesize_stream(
        text=phraze,
        voice='alena',emotion='good', format='lpcm', sampleRateHertz=sample_rate
    )
    pyaudio_play_audio_function(audio_data, sample_rate=sample_rate)

def recognize_speech():
    print("Audio ready to listen")
    # Записываем аудио продолжительностью 3 секунды
    audio_data = record_audio(3, sample_rate)
    print("Record finish. Trying to understand you")

    # Отправляем на распознавание
    text = recognizeShortAudio.recognize(
        audio_data, format='lpcm', sampleRateHertz=sample_rate)
    print(text)
    return text
