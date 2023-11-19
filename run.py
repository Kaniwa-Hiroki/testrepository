import wave
from scipy.signal import hamming, medfilt, butter, lfilter
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import librosa
import pyaudio
import pickle
import time
from tqdm import tqdm
import sys
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

class AudioManager:
    # コンストラクタ
    # サンプリングレート、フレーム数を設定する
    def __init__(self, 
                 sampling_rate=44100, 
                 frames_per_buffer=1024, 
                 channels=1):
        self._sampling_rate = sampling_rate
        self._frames_per_buffer = frames_per_buffer
        self._channels = channels

    def __del__(void):
        pass
        
    # マイクで拾った音声を取得可能な状態にする関数
    def setup_audio_capture(self):
        self._audio = pyaudio.PyAudio()
        self._audio_stream = self._audio.open(rate=self._sampling_rate, 
                                              channels=self._channels,
                                              format=pyaudio.paInt16,
                                              input=True,
                                              frames_per_buffer=self._frames_per_buffer)
    
    # マイクで拾った音声の取得を終了する関数
    def cleanup_audio_capture(self):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._audio.terminate()
    
    # マイクで拾った音声を取得する関数
    # 音声が取得可能でない場合は0を返す
    def get_audio_capture(self):
        if not self._audio_stream.is_active():
            return np.empty(0)
        
        raw_audio_data = self._audio_stream.read(self._frames_per_buffer)
        array_data = np.frombuffer(raw_audio_data, dtype=np.int16)
    
        return array_data


# バンドパスフィルタの設計
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# バンドパスフィルタの適用
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def load_model(clf_path):
    with open(clf_path, mode='rb') as f:
        list_models = pickle.load(f)
    return list_models

frames = 1024
lowcut = 20
highcut = 5000
CHUNK = 1024
RATE = 44100
window = hamming(CHUNK)
scaler = MinMaxScaler()

list_models = load_model("list_models.pkl")
audio_manager = AudioManager(frames_per_buffer=frames)
audio_manager.setup_audio_capture()

list_time = []
num_is_drone = 0
count = 1000

#for i in tqdm(range(count)):
for i in range(count):
    try:
        start = time.time()
        audio_data = audio_manager.get_audio_capture()
        array_data = butter_bandpass_filter(audio_data, lowcut, highcut, RATE)
        array_data = medfilt(audio_data, kernel_size=3)
        spectrum = np.abs(np.fft.fft(array_data * window))[:CHUNK] / CHUNK
        normalized_spectrum = scaler.fit_transform(spectrum.reshape(-1, 1))
        
        end = time.time()
        time_diff = end - start

        list_predict = []
        for model in list_models:
            list_predict.append(model[1].predict([normalized_spectrum.reshape(-1)]))

        sys.stdout.write("\r"+str(sum(list_predict)/len(list_predict))+str(list_predict)+" time="+str(time_diff))
        sys.stdout.flush()
        if sum(list_predict) == 0:
            print("ドローン離陸!")

        if sum(list_predict) == 0:
            num_is_drone += 1

        list_time.append(time_diff)

    except KeyboardInterrupt: ## ctrl + c
        break

print("")
print(f"ループあたりの実行時間　平均: {sum(list_time) / len(list_time)}s, 最大: {max(list_time)}s")
print(f"分類結果　試行回数: {count}回, ドローン有回数: {num_is_drone}回, ドローン有確率: {(num_is_drone / count) * 100}%")