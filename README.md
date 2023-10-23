# import関係
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import pickle
import pyaudio
import librosa
import numpy as np

class AudioManager:
    #コンストラクタ
    # サンプリングレート、フレーム数を設定する
    def __init__(self, 
                 sampling_rate: int = 44100, 
                 frames_per_buffer: int = 8192, 
                 channels: int = 1):
        self.__sampling_rate = sampling_rate
        self.__frames_per_buffer = frames_per_buffer
        self.__channels = channels

    def __del__(void):
        pass
        
    # マイクで拾った音声を取得可能な状態にする関数
    def setup_audio_capture(self):
        self.__audio = pyaudio.PyAudio()
        self.__audio_stream = self.__audio.open(rate=self.__sampling_rate, 
                                               channels=self.__channels,
                                               format=pyaudio.paFloat32,
                                               input=True,
                                               frames_per_buffer=self.__frames_per_buffer)
    
    # マイクで拾った音声の取得を終了する関数
    def cleanup_audio_capture(self):
        self.__audio_stream.stop_stream()
        self.__audio_stream.close()
        self.__audio.terminate()
    
    # マイクで拾った音声を取得する関数
    # 音声が取得可能でない場合 or 取得した音声が無音の場合は0を返す
    # 引数 threshold: 有音と判定する閾値
    def get_audio_capture(self, threshold=0.05) -> np.ndarray:
        if not self.__audio_stream.is_active():
            return np.empty(0)
        
        raw_audio_data = self.__audio_stream.read(self.__frames_per_buffer)
        audio_data = np.frombuffer(raw_audio_data, dtype='float32')
        if max(audio_data) < threshold: ## 無音判定
            return np.empty(0)
    
        return audio_data
    
    # 音データを加工する関数
    def preprocess_audio_data(self, audio_data) -> np.ndarray:
        return librosa.feature.mfcc(y=audio_data, sr=self.__sampling_rate)[1:]

    # 録音データを読込む関数
    def load_audio_data(self, data_path: str) -> np.ndarray:
        audio_data, _ = librosa.load(data_path, sr=self.__sampling_rate)
        return audio_data


class AudioClassifier:
    def __init__(self):
        pass

    def __del__(self):
        pass
        
    # モデルの学習に使用するラベル付きのデータセットを作成する関数
    def create_labeled_dataset(self, list_data, test_size=0.3):
        list_labeled_data = []
        for idx in range(len(list_data)):
            #list_labeled_data = list_labeled_data + [item + [idx] for item in list_data[idx]]
            list_labeled_data = list_labeled_data + [list(pair) + [idx] for pair in zip(*list_data[idx])]
        array_labeled_data = np.array(list_labeled_data)
        np.random.shuffle(array_labeled_data)
        
        test_num = int(len(array_labeled_data) * test_size)
        self.__X_train = array_labeled_data[test_num:, :-1]
        self.__y_train = array_labeled_data[test_num:, -1]
        self.__X_test = array_labeled_data[:test_num, :-1]
        self.__y_test = array_labeled_data[:test_num, -1]
        print(self.__X_train[0].shape)

    # 教師データを用いて2値分類のモデルを学習する関数
    def fit_model(self):
        self.__clf = svm.SVC()
        self.__clf.fit(self.__X_train, self.__y_train)

    def check_model(self):
        y_predict = self.__clf.predict(self.__X_test)
        print("データ数=" + str(len(self.__y_test)))
        print("正解率=%f"%accuracy_score(self.__y_test, y_predict))
    
    # モデルを保存する関数
    def save_model(self, clf_path):
        with open(clf_path, mode='wb') as f:
            pickle.dump(self.__clf, f, protocol=2)
    
    # 保存されたモデルを読込む関数
    def load_model(self, clf_path):
        with open(clf_path, mode='rb') as f:
            self.__clf = pickle.load(f)
    
    # 学習済みモデルを利用して2値分類を行う関数
    def classify_audio_data(self, data):
        return self.__clf.predict([data])

# メイン関数（音声分類）
def main_classify():
    clf_path = 'clf.pkl'
    
    audio_manager = AudioManager()
    audio_classifier = AudioClassifier()
    audio_classifier.load_model(clf_path)
    audio_manager.setup_audio_capture()
    while True:
        try:
            start = time.time()
            audio_data = audio_manager.get_audio_capture()
            if len(audio_data) == 0:
                continue
            mfcc = preprocess_audio_data(audio_data)
            data = np.mean(mfcc, axis=1, keepdims=True).reshape(-1)
            
            predict = audio_classifier.classify_audio_data(data)
            end = time.time()
            time_diff = end - start
            ## 推測結果
            print(predict[0], time_diff)
    
        except KeyboardInterrupt: ## ctrl + c
            break

# メイン関数（学習）
def main_fit():
    data1_path = 'data1.wav'
    data2_path = 'data2.wav'
    clf_path = 'clf.pkl'
    audio_manager = AudioManager()
    audio_data1 = audio_manager.load_audio_data(data1_path)
    audio_data2 = audio_manager.load_audio_data(data2_path)
    mfcc_audio_data1 = audio_manager.preprocess_audio_data(audio_data1)
    mfcc_audio_data2 = audio_manager.preprocess_audio_data(audio_data2)
    audio_classifier = AudioClassifier()
    audio_classifier.create_labeled_dataset([mfcc_audio_data1, mfcc_audio_data2])
    audio_classifier.fit_model()
    audio_classifier.check_model()
    audio_classifier.save_model(clf_path)

if __name__ == "__main__":
    #main_fit()
    main_classify()
