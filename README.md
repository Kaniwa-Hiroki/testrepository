from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import pyaudio
import librosa
import numpy as np

def data_loader(data_path, default_sr=44100):
     return librosa.load(data_path, sr=default_sr)
     
def get_mfcc(data, sr):
    return librosa.feature.mfcc(y=data, sr=sr)

## mfccデータを整える	
def get_train_test_data(list_mfccs, test_size=0.2):
	data = []
	list_data, list_label = [], []
	for i in range(len(list_mfccs)):
		for j in range(len(list_mfccs[i][0])):	
			data.append([list_mfccs[i][1:,j],i])
	np.random.shuffle(data)
	for i in range(len(data)):
		list_data.append(data[i][0])
		list_label.append(data[i][1])

	test_num = int(len(list_data) * test_size)

	X_test = list_data[:test_num]
	y_test = list_label[:test_num]
	X_train = list_data[test_num:]
	y_train = list_label[test_num:]
    
	return X_train, X_test, y_train, y_test

def fit_svm_clf(clf_name, X_train, y_train):
    #start = time.time()
    clf = svm.SVC()
    clf.fit(train_data, train_label)
    #end = time.time()
    #time_diff = end - start
    #print(time_diff)
    
    with open(clf_name, mode='wb') as f:
        pickle.dump(clf,f,protocol=2)
    
    return clf

data1_path = "data1.wav"
data2_path = "data2.wav"
svm_clf_name = "svm_clf.pkl"
test_size = 0.2
random_state = 7

data1, sr1 = data_loader(data1_path)
data2, sr2 = data_loader(data2_path)

mfcc1 = get_mfcc(data1, sr1)
mfcc2 = get_mfcc(data2, sr2)

X_train, X_test, y_train, y_test = \
    get_train_test_data([mfcc1, mfcc2], test_size=test_size)

svm_clf = fit_svm_clf(svm_clf_name, X_train, y_train)
predict_label = clf.predict(X_test)
print("テストデータ数:%d"%len(y_test))
print("正解率 = %f"%accuracy_score(y_test, predict_label))

//////////////////////////////////////////////////////////////////////////

import pickle
import librosa
import pyaudio
import numpy as np

BUFSIZE = 8192
SAMPLING_RATE = 44100 ## サンプリングレート

## pcの標準音声入力から音声取得
audio = pyaudio.PyAudio()
stream = audio.open( rate=SMPL_RATE,
				channels=1,
				format=pyaudio.paFloat32,
				input=True,
				frames_per_buffer=BUFSIZE)

status = 0

## 分類器を取り込む
svm_clf_name = "svm_clf.pkl"
with open(svm_clf_name, mode='rb') as f:
    clf = pickle.load(f)

print("start")
while True:
    try:
        start = time.time()
        audio_data=stream.read(BUFSIZE)
        data=np.frombuffer(audio_data,dtype='float32')
        if max(data) < 0.05: ## 無音判定
            status = 0
            continue

        ## mfccを算出def get_mfcc(data, sr):
        mfcc = get_mfcc(data, SAMPLING_RATE)
        mfcc_mean = np.array([])
        for i in range(1,len(mfcc)):
            mfcc_mean = np.append(mfcc_mean,np.mean(mfcc[i]))

        ## 分類器に入れる
        predicted_value= clf.predict([mfcc_mean])
        end = time.time()
        time_diff = end - start
        ## 推測結果
        print(predicted_value[0], time_diff)

    except KeyboardInterrupt: ## ctrl + c
        break

## 後始末
stream.stop_stream()
stream.close()
audio.terminate()
