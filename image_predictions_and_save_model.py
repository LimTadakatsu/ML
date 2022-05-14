import numpy as np
from keras.utils import np_utils #匯入 np_utils模組 利用to_categorical轉換
np.random.seed(10)
from keras.datasets import mnist#先載入模組
import matplotlib.pyplot as plt #Pyplot 是 Matplotlib 的子庫，提供了和 MATLAB 類似的繪圖 API
from keras.models import Sequential
from keras.layers import Dense


def show_image(image): #定義秀出照片
    fig = plt.gcf()
    fig.set_size_inches(2,2) #照片size 2*2吋
    plt.imshow(image, cmap="binary") #白底黑字顯示 #binary二進制
    plt.show()

def show_images_labels_predictions(images,labels,
                                  classes_x,start_id,num=10):
    plt.gcf().set_size_inches(12, 14)
    if num>25: num=25 #
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[start_id], cmap='binary') #顯示黑白圖片
        
        
        if( len(classes_x) > 0 ) :  # 有 AI 預測結果資料, 才在標題顯示預測結果
            title = 'ai = ' + str(classes_x[start_id]) 
            title += (' (o)' if classes_x[start_id]==labels[start_id] else ' (x)')   # 預測正確顯示(o), 錯誤顯示(x)
            title += '\nlabel = ' + str(labels[start_id])
       
        else :
            title = 'label = ' + str(labels[start_id])  # 沒有 AI 預測結果資料, 只在標題顯示真實數值
            
        # X, Y 軸不顯示刻度    
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
    plt.show()

#如mnist load.py那邊一樣





#建立訓練資料和測試資料，包括訓練特徵集、訓練標籤和測試特徵集、測試標籤	
(train_feature, train_label),\
(test_feature, test_label) = mnist.load_data()

train_feature_vector =train_feature.reshape(len(train_feature), 784).astype('float32') #將圖片轉成1維向量
test_feature_vector = test_feature.reshape(len( test_feature), 784).astype('float32')

train_feature_normalize = train_feature_vector/255 #Features 特徵值標準化 除以255讓數字變成0~1之間的浮點數 增加效率與準確度
test_feature_normalize = test_feature_vector/255


train_label_onehot = np_utils.to_categorical(train_label) #label 轉換為 One-Hot Encoding 編碼
test_label_onehot = np_utils.to_categorical(test_label)


model = Sequential() #建立模型

from keras.layers import Dense #使用keras中的 dense(核心網路層)

model.add(Dense(units=256, # 隱藏層：256
                input_dim=784, #輸入層：784
                kernel_initializer='normal', 
                activation='relu'))
model.add(Dense(units=10, #輸出層：10
                kernel_initializer='normal', 
                activation='softmax'))


model.compile(loss='categorical_crossentropy', #定義訓練方式 需要建立loss損失函式 optimizer 最佳化方法 metrics評估準確率的方式 
              optimizer='adam', metrics=['accuracy'])

train_history =model.fit(x=train_feature_normalize, #x=特徵值  y=標籤 validation_split=驗證資料百分比 epochs=訓練次數, batch_size=每次讀取數量,verbose=顯示訓練過程 0不顯示 1 顯示 2簡易顯示
                         y=train_label_onehot,validation_split=0.2, 
                         epochs=10, batch_size=200,verbose=2)


scores = model.evaluate(test_feature_normalize, test_label_onehot) #評估準確率

print('\n準確率=',scores[1])



prediction=model.predict(test_feature_normalize) #預測的寫法之二
prediction=np.argmax(prediction,axis=1)

# prediction=model.predict_classes(test_feature_normalize) #預測(預測的一種寫法，不知道為何不能在這使用) #經查用tensorflow 2.5.0v 可以使用但有提示要改寫法了



show_images_labels_predictions(test_feature,test_label,prediction,0) #秀出10張圖片 顯示圖像、預測值、真實值 從line16撈資料欄

model.save("firstmodel.h5")
print("儲存完畢")