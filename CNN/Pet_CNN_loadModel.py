import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
np.random.seed(10)

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
    
imagesavepath='Cat_Dog_Dataset/'
try:
    test_feature=np.load(imagesavepath+'test_feature.npy')
    test_label=np.load(imagesavepath+'test_label.npy')       
    print("載入 *.npy 檔!") 
    
    #將 Features 特徵值換為 圖片數量*40*40*3 的 4 維矩陣
    test_feature_vector = test_feature.reshape(len( test_feature), 40,40,3).astype('float32')
    
    #Features 特徵值標準化
    test_feature_normalize = test_feature_vector/255
    
    #從 HDF5 檔案中載入模型
    print("載入模型 Pet_cnn_model.h5")
    model = load_model('Pet_cnn_model.h5')
        
    #預測
    prediction=model.predict(test_feature_normalize) #預測的寫法之二
    prediction=np.argmax(prediction,axis=1)
    
    #顯示圖像、預測值、真實值 
    show_images_labels_predictions(test_feature,test_label,prediction,0)
except:
    print(".npy 檔未建立!")     