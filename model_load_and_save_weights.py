import numpy as np
from keras.utils import np_utils #匯入 np_utils模組 利用to_categorical轉換
np.random.seed(10)
from keras.datasets import mnist#先載入模組
import matplotlib.pyplot as plt #Pyplot 是 Matplotlib 的子庫，提供了和 MATLAB 類似的繪圖 API
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model #已儲存的模組
import glob
import cv2


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



#建立訓練資料和測試資料，包括訓練特徵集、訓練標籤和測試特徵集、測試標籤	
# (train_feature, train_label),\
# (test_feature, test_label) = mnist.load_data()

files=glob.glob("imagedata\*.jpg")

test_feature=[]
test_label=[]

for file in files:
    img=cv2.imread(file)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰階
    _, img=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)#反向黑白
    test_feature.append(img)
    label=file[10:11] #EX "imagedata\1.jpg"第十個字元1為label
    test_label.append(int(label))

test_feature=np.array(test_feature) #串列轉為矩陣
test_label=np.array(test_label)


# test_label_vector =test_label.reshape(len(test_label), 784).astype('float32') #將圖片轉成1維向量
test_feature_vector = test_feature.reshape(len(test_feature), 784).astype('float32')
#測試是否成功轉為一維向量
print(test_feature_vector.shape)
# train_feature_normalize = train_feature_vector/255 #Features 特徵值標準化 除以255讓數字變成0~1之間的浮點數 增加效率與準確度
test_feature_normalize = test_feature_vector/255


# # train_label_onehot = np_utils.to_categorical(train_label) #label 轉換為 One-Hot Encoding 編碼
# test_label_onehot = np_utils.to_categorical(test_label)

#從HDF5檔案中載入模型
print("載入模型")
model=load_model("firstmodel.h5")
print("載入成功")

#預測
# prediction=model.predict_classes(test_feature_normalize) #還是不行用的寫法QQ
prediction=model.predict(test_feature_normalize) #預測的寫法之二cls
prediction=np.argmax(prediction,axis=1)

show_images_labels_predictions(test_feature,test_label,prediction,0) #秀出圖片

model.save_weights("firstweights")
print("儲存成功")