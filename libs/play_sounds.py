import pyaudio
import time

import os 

directory = os.getcwd()
#directory="/home/jetson/Documents/VSOSH_region_eg_ed"
os.chdir(directory)

def pyaudio_play_audio_function(audio_data, num_channels=1, 
                                sample_rate=16000, chunk_size=4000) -> None:
    """
    Воспроизводит бинарный объект с аудио данными в формате lpcm (WAV)
    
    :param bytes audio_data: данные сгенерированные спичкитом
    :param integer num_channels: количество каналов, спичкит генерирует 
        моно дорожку, поэтому стоит оставить значение `1`
    :param integer sample_rate: частота дискретизации, такая же 
        какую вы указали в параметре sampleRateHertz
    :param integer chunk_size: размер семпла воспроизведения, 
        можно отрегулировать если появится потрескивание
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=num_channels,
        rate=sample_rate,
        output=True,
        frames_per_buffer=chunk_size
    )
    #Прочитать и может переделать на callback https://github.com/raspberrypi/linux/issues/994, https://people.csail.mit.edu/hubert/pyaudio/
    # diffrent libs to play sounds: https://realpython.com/playing-and-recording-sound-python/#pyaudio
    #можно попробовать использовать https://pypi.org/project/sounddevice/
    #или alsaaudio/pyalsaaudio https://github.com/larsimmisch/pyalsaaudio/blob/master/playwav.py
    try:
        for i in range(0, len(audio_data), chunk_size):
            stream.write(audio_data[i:i + chunk_size])
    finally:
        # Wait the stream end                                                                                                              
        time.sleep(1.0)
        #stop and kill stream
        stream.stop_stream()
        stream.close()
        p.terminate()

sample_rate = 16000 # частота дискретизации должна 
                    # совпадать при синтезе и воспроизведении


names={0: 'Банан', 1: 'Груша', 2: 'Ананас', 3: 'Клубника'}

colors={0:'оранжевый',1:'красный',2:'фиолетовый',3:'голубой',4:'зеленый', 5:'желтый'}

phrazes={"found":"Десятка!", "listen":"Слушаю Вас!","robot_win":"УРа-УРА!! робот выиграл","nobody":"В этот раз тебе повезло! Ничья!","human_win":"как тебе это удалось, о мой повелитель? ты победил!"}



def Say_object_class(index):
    # Читаем файл
    print('frukt_class_'+str(index)+'.wav', names[index])
    with open("sounds/frukt_class_"+str(index)+'.wav', 'rb') as f:
       audio_data = f.read()

    pyaudio_play_audio_function(audio_data, sample_rate=sample_rate)

def Say_object_color(color):
    color=color.lower()
    # Поиск ключа по значению
    index = list(colors.values()).index(color)
    # Читаем файл
    print('color_'+str(index)+'.wav', colors[index])
    with open("sounds/color_"+str(index)+'.wav', 'rb') as f:
       audio_data = f.read()

    pyaudio_play_audio_function(audio_data, sample_rate=sample_rate)

def Say_phraze(template):
    # Читаем файл
    with open("sounds/"+str(template)+'.wav', 'rb') as f:
       audio_data = f.read()
    pyaudio_play_audio_function(audio_data, sample_rate=sample_rate)

#Say_phraze("found")
#Say_object_class(3)
#Say_object_color("Зеленый")
#phraza="Слушаю Вас"
# if score_robot>score_robot:
#     Say_phraze("robot_win")
# else if score_robot==score_human:
#     Say_phraze("nobody")
# else:
#     Say_phraze("human_win")