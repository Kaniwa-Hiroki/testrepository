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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


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

def save_model(list_models, clf_path):
    with open(clf_path, mode='wb') as f:
        pickle.dump(list_models, f, protocol=2)

wav_file_path1 = 'data_p2.wav'
wav_file_path2 = 'data_n2.wav'
list_wf = [wave.open(wav_file_path1, 'rb'), wave.open(wav_file_path2, 'rb')]

frames = 1024
lowcut = 20
highcut = 5000
CHUNK = 1024
RATE = 44100
window = hamming(CHUNK)
scaler = MinMaxScaler()

list_data = []

for i in range(len(list_wf)):
    list_data.append([])
    while True:
        data = list_wf[i].readframes(frames)
        array_data = np.frombuffer(data, dtype=np.int16)
        array_data = butter_bandpass_filter(array_data, lowcut, highcut, RATE) #
        if len(array_data) < len(window):
            break
            
        array_data = medfilt(array_data, kernel_size=3)
        spectrum = np.abs(np.fft.fft(array_data * window))[:CHUNK] / CHUNK
        normalized_spectrum = scaler.fit_transform(spectrum.reshape(-1, 1))
        
        list_data[i].append(normalized_spectrum)

array1 = np.array(list_data[0])
array2 = np.array(list_data[1])

combined_array = np.concatenate([array1, array2], axis=0)

labels = np.concatenate([np.zeros((array1.shape[0], 1, 1)), np.ones((array2.shape[0], 1, 1))], axis=0)
combined_array_with_labels = np.concatenate([combined_array, labels], axis=1)
final_result = np.squeeze(combined_array_with_labels, axis=-1)

np.random.seed(42)
np.random.shuffle(final_result)

test_size = 0.3
test_num = int(len(final_result) * test_size)
_X_train = final_result[test_num:, :-1]
_y_train = final_result[test_num:, -1]
_X_test = final_result[:test_num, :-1]
_y_test = final_result[:test_num, -1]

list_models = []
list_models.append(["ExtraTrees", ExtraTreesClassifier(bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200)])
list_models.append(["HistGradientBoosting", HistGradientBoostingClassifier(learning_rate=0.1, max_depth=10, max_iter=300)])
list_models.append(["RandomForest", RandomForestClassifier(bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200)])

for model in list_models:
    model[1].fit(_X_train, _y_train)
    
    y_predict = model[1].predict(_X_test)

    print(model[0])
    print(f"正解率: {accuracy_score(_y_test, y_predict)}")
    print(f"適合率: {precision_score(_y_test, y_predict)}")
    print(f"再現率: {recall_score(_y_test, y_predict)}")
    print(f"F1スコア: {f1_score(_y_test, y_predict)}")
    print("")

save_model(list_models, "list_models.pkl")