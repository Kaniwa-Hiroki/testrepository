# import関係
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import time
import pickle
import pyaudio
import librosa
import numpy as np
from collections import deque

class AudioManager:
    #コンストラクタ
    # サンプリングレート、フレーム数を設定する
    def __init__(self, 
                 sampling_rate: int = 44100, 
                 frames_per_buffer: int = 8192, 
                 channels: int = 1):
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
                                               format=pyaudio.paFloat32,
                                               input=True,
                                               frames_per_buffer=self._frames_per_buffer)
    
    # マイクで拾った音声の取得を終了する関数
    def cleanup_audio_capture(self):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._audio.terminate()
    
    # マイクで拾った音声を取得する関数
    # 音声が取得可能でない場合 or 取得した音声が無音の場合は0を返す
    # 引数 threshold: 有音と判定する閾値
    def get_audio_capture(self, threshold=0.02) -> np.ndarray:
        if not self._audio_stream.is_active():
            return np.empty(0)
        
        raw_audio_data = self._audio_stream.read(self._frames_per_buffer)
        audio_data = np.frombuffer(raw_audio_data, dtype='float32')
        if max(abs(audio_data)) < threshold: ## 無音判定
            return np.empty(0)
    
        return audio_data
    
    # 音データを加工する関数
    def preprocess_audio_data(self, audio_data) -> np.ndarray:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self._sampling_rate)[1:13]
        mean = np.mean(mfccs, axis=0)
        std = np.std(mfccs, axis=0)
        standardized_mfccs = (mfccs - mean) / std
        return standardized_mfccs

    # 録音データを読込む関数
    def load_audio_data(self, data_path: str) -> np.ndarray:
        audio_data, _ = librosa.load(data_path, sr=self._sampling_rate)
        return audio_data

import scipy

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
        self._X_train = array_labeled_data[test_num:, :-1]
        self._y_train = array_labeled_data[test_num:, -1]
        self._X_test = array_labeled_data[:test_num, :-1]
        self._y_test = array_labeled_data[:test_num, -1]
        return self._X_train, self._y_train, self._X_test, self._y_test

    # 教師データを用いて2値分類のモデルを学習する関数
    def fit_model(self):
        self._clf = RandomForestClassifier()
        self._clf.fit(self._X_train, self._y_train)

    def fit_list_model(self):
        self._list_clf = []
        self._list_clf.append(["LogisticRegression", LogisticRegression()])
        self._list_clf.append(["SVC", SVC(probability=True)])
        self._list_clf.append(["RandomForestClassifier", RandomForestClassifier()])
        self._list_clf.append(["KNeighborsClassifier", KNeighborsClassifier()])

        for clf in self._list_clf:
            clf[1].fit(self._X_train, self._y_train)

    def check_list_model(self):
        for clf in self._list_clf:
            y_predict = clf[1].predict(self._X_test)
            print(clf[0])
            print(f"正解率: {accuracy_score(self._y_test, y_predict)}")
            print(f"適合率: {precision_score(self._y_test, y_predict)}")
            print(f"再現率: {recall_score(self._y_test, y_predict)}")
            print(f"F1スコア: {f1_score(self._y_test, y_predict)}")

    def check_model(self):
        y_predict = self._clf.predict(self._X_test)
        #y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
        print(f"正解率: {accuracy_score(self._y_test, y_predict)}")
        print(f"適合率: {precision_score(self._y_test, y_predict)}")
        print(f"再現率: {recall_score(self._y_test, y_predict)}")
        print(f"F1スコア: {f1_score(self._y_test, y_predict)}")

    def fit_model2(self):
        train_data = lgb.Dataset(self._X_train, label=self._y_train)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "device": "gpu"  # GPUを使用するための設定
        }
        num_round = 100
        model = lgb.train(params, train_data, num_round)
        model.save_model("best_model", num_iteration=model.best_iteration)
        self._model = lgb.Booster(model_file="best_model")

    def check_model2(self):
        # 精度（precision）を計算
        y_pred = self._model.predict(self._X_test)
        y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
        print(f"適合率: {precision_score(self._y_test, y_pred_binary)}")
        print(f"再現率: {recall_score(self._y_test, y_pred_binary)}")
        print(f"F1スコア: {f1_score(self._y_test, y_pred_binary)}")
        print(self._X_train[:10])
    
    # モデルを保存する関数
    def save_model(self, clf_path):
        with open(clf_path, mode='wb') as f:
            pickle.dump(self._clf, f, protocol=2)
    
    # 保存されたモデルを読込む関数
    def load_model(self, clf_path):
        with open(clf_path, mode='rb') as f:
            self._clf = pickle.load(f)

    # 保存されたモデルを読込む関数
    def load_model2(self, clf_path):
        self._model = lgb.Booster(model_file="best_model")
    
    # 学習済みモデルを利用して2値分類を行う関数
    def classify_audio_data(self, data):
        return self._clf.predict([data]), self._clf.predict_proba([data])
    
    def classify_audio_data_list(self, data):
        list_results = []
        for clf in self._list_clf:
            list_results.append([clf[0], clf[1].predict([data]), clf[1].predict_proba([data])])
            list_results.append(clf[1].predict([data]))
        return list_results

    # 学習済みモデルを利用して2値分類を行う関数
    def classify_audio_data2(self, data):
        y_pred = self._model.predict(data)
        y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
        return y_pred, y_pred_binary

    def test(self):
        pass

# メイン関数（音声分類）
def main_classify():
    clf_path = 'clf.pkl'
    len_drone = 5
    
    audio_manager = AudioManager()
    audio_classifier = AudioClassifier()
    audio_classifier.load_model(clf_path)
    audio_manager.setup_audio_capture()
    queue = deque(maxlen=len_drone)
    for i in range(len_drone):
        queue.append(0)
        
    while sum(queue) <= len_drone:
        try:
            start = time.time()
            audio_data = audio_manager.get_audio_capture()
            if len(audio_data) == 0:
                continue
            mfcc = audio_manager.preprocess_audio_data(audio_data)
            data = np.mean(mfcc, axis=1, keepdims=True).reshape(-1)
            
            #queue.append(audio_classifier.classify_audio_data(scipy.stats.zscore(data))[0])
            print(audio_classifier.classify_audio_data(data))
            queue.appendleft(int(audio_classifier.classify_audio_data(data)[0]))
            end = time.time()
            time_diff = end - start
            ## 推測結果
            print(queue[0], list(queue), time_diff)
    
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
    pass
    #main_fit()
    #main_classify()
