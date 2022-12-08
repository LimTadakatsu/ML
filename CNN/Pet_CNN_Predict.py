import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
from keras.models import load_model
import glob,cv2

def show_images_labels_predictions(images,labels,
                                  classes_x,start_id,num=10):
    plt.gcf().set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        #顯示彩色圖片
        ax.imshow(images[start_id])
        
        # 有 AI 預測結果資料, 才在標題顯示預測結果
        if( len(classes_x) > 0 ) :
            title = 'ai = ' + str(classes_x[start_id])
            # 預測正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if classes_x[start_id]==labels[start_id] else ' (x)') 
            title += '\nlabel = ' + str(labels[start_id])
        # 沒有 AI 預測結果資料, 只在標題顯示真實數值
        else :
            title = 'label = ' + str(labels[start_id])
            
        # X, Y 軸不顯示刻度    
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
    plt.show()
    
#建立測試特徵集、測試標籤	    
files = glob.glob("imagedata\*.jpg" )
test_feature=[]
test_label=[]
dict_labels = {"Cat":0, "Dog":1}
size = (40,40) #由於原始資料影像大小不一，因此制定一個統一值
for file in files:
    img=cv2.imread(file) 
    img = cv2.resize(img, dsize=size)       
    test_feature.append(img)
    label=file[10:13]  # "imagedata\Cat1.jpg" 第10-12個字元 Cat為 label
    test_label.append(dict_labels[label])
   
test_feature=np.array(test_feature) # 串列轉為矩陣 
test_label=np.array(test_label)     # 串列轉為矩陣

#將 Features 特徵值換為 圖片數量*80*80*3 的 4 維矩陣
test_feature_vector =test_feature.reshape(len(test_feature), 40,40,3).astype('float32')

#Features 特徵值標準化
test_feature_normalize = test_feature_vector/255

try:
    #從 HDF5 檔案中載入模型
    print("載入模型 Pet_cnn_model.h5")
    model = load_model('Pet_cnn_model.h5')
        
    #預測
    prediction=model.predict(test_feature_normalize) #預測的寫法之二
    prediction=np.argmax(prediction,axis=1)
    
    #顯示圖像、預測值、真實值 
    show_images_labels_predictions(test_feature,test_label,prediction,0,len(test_feature))
except:
    print("模型未建立!")