from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Dropout
from keras.callbacks import ModelCheckpoint
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import numpy as np
(X_train,y_train),(X_test,y_test)=mnist.load_data()


print(len(X_test))
#fig=plt.figure(figsize=(20,20))
#for i in range(6):
    #ax=fig.add_subplot(1,6,i+1,xticks=[],yticks=[])
    #ax.imshow(X_train[i],cmap='gray')
    
    
    
X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255    

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

model=Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(720,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(720,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])

#checkpoint=ModelCheckpoint(filepath='mnist.model.best.hdf5',save_best_only=True,verbose=1)
#hist=model.fit(X_train,y_train,batch_size=128,epochs=10,validation_split=0.2,callbacks=[checkpoint],verbose=1,shuffle=True)
model.load_weights('mnist.model.best.hdf5')
score=model.evaluate(X_test,y_test,verbose=0)
print(score[1])