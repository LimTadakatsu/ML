import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
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
            title = 'ai = ' + str(classes_x[i])
            # 預測正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if classes_x[i]==labels[i] else ' (x)') 
            title += '\nlabel = ' + str(labels[i])
        # 沒有 AI 預測結果資料, 只在標題顯示真實數值
        else :
            title = 'label = ' + str(labels[i])
            
        # X, Y 軸不顯示刻度    
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
    plt.show()

imagesavepath='Cat_Dog_Dataset/'
try:    
    train_feature=np.load(imagesavepath+'train_feature.npy')  
    test_feature=np.load(imagesavepath+'test_feature.npy')  
    train_label=np.load(imagesavepath+'train_label.npy')      
    test_label=np.load(imagesavepath+'test_label.npy')       
    print("載入 *.npy 檔!") 

    #將 Features 特徵值換為 圖片數量*80*80*3 的 4 維矩陣
    train_feature_vector =train_feature.reshape(len(train_feature), 40,40,3).astype('float32')
    test_feature_vector = test_feature.reshape(len( test_feature), 40,40,3).astype('float32')
    
    #Features 特徵值標準化
    train_feature_normalize = train_feature_vector/255
    test_feature_normalize = test_feature_vector/255
    
    #label 轉換為 One-Hot Encoding 編碼
    train_label_onehot = np_utils.to_categorical(train_label)
    test_label_onehot = np_utils.to_categorical(test_label)
    
    #建立模型
    model = Sequential()
    #建立卷積層1
    model.add(Conv2D(filters=10, 
                      kernel_size=(5,5),
                      padding='same',
                      input_shape=(40,40,3), 
                      activation='relu'))
    
    #建立池化層1
    model.add(MaxPooling2D(pool_size=(2, 2))) #(10,40,40)
    
    # Dropout層防止過度擬合，斷開比例:0.1
    model.add(Dropout(0.1))    
    
    #建立卷積層2
    model.add(Conv2D(filters=20, 
                      kernel_size=(5,5),  
                      padding='same',
                      activation='relu'))
    
    #建立池化層2
    model.add(MaxPooling2D(pool_size=(2, 2))) #(20,20,20)
    
    # Dropout層防止過度擬合，斷開比例:0.2
    model.add(Dropout(0.2))
    
    #建立平坦層：20*20*20=8000 個神經元
    model.add(Flatten()) 
    
    #建立隱藏層
    model.add(Dense(units=512, activation='relu'))
    
    #建立輸出層
    model.add(Dense(units=2,activation='softmax'))
    
    # 這些訓練會累積，準確會愈來愈高
    try:
        model.load_weights("Pet_cnn_model.weight")
        print("載入模型參數成功，繼續訓練模型!")
    except :    
        print("載入模型失敗，開始訓練一個新模型!")
    
    #定義訓練方式
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    
    #以(train_feature_normalize,train_label_onehot)資料訓練，
    #訓練資料保留 20% 作驗證,訓練2次、每批次讀取200筆資料，顯示簡易訓練過程
    train_history =model.fit(x=train_feature_normalize,
                             y=train_label_onehot,validation_split=0.2, 
                             epochs=2, batch_size=200,verbose=2)
    #評估準確率
    scores = model.evaluate(test_feature_normalize, test_label_onehot)
    print('\n準確率=',scores[1])
        
    #預測
    prediction=model.predict(test_feature_normalize) #預測的寫法之二
    prediction=np.argmax(prediction,axis=1)
    
    # 儲存模型
    model.save('Pet_cnn_model.h5')
    print("Pet_cnn_model.h5 模型儲存完畢!")
    model.save_weights("Pet_cnn_model.weight")
    print("Pet_cnn_model.weight 模型參數儲存完畢!")
    
    del model    
    
    #顯示圖像、預測值、真實值 
    show_images_labels_predictions(test_feature,test_label,prediction,0)
except:
    print(".npy 檔未建立!")      