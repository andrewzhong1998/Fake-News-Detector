from keras.models import *
from keras.layers import *
from keras.optimizers import *

def load_dataset():
    X_Test = np.load('data/test_features.npy')
    Y_Test = np.load('data/test_labels.npy')
    X_Valid = np.load('data/validation_features.npy')
    Y_Valid = np.load('data/validation_labels.npy')
    X_Train = np.stack((np.load('data/training_features1.npy'), np.load('data/training_features2.npy')), axis=0)
    Y_Train = np.stack((np.load('data/training_labels1.npy'), np.load('data/training_labels2.npy')), axis=0)
    Y_Test = Y_Test.reshape((2000,1))
    Y_Valid = Y_Valid.reshape((2000,1))
    X_Train = X_Train.reshape((16000,209429))
    Y_Train = Y_Train.reshape((16000,1))
    return X_Train, Y_Train, X_Valid, Y_Valid, X_Test, Y_Test

def model(input_size=(209429,)):
    inputs = Input(input_size)
    X1 = Dense(30, activation='relu')(inputs)
    X2 = Dense(20, activation='relu')(X1)
    X3 = Dense(20, activation='relu')(X2)
    X4 = Dense(10, activation='relu')(X3)
    X5 = Dense(10, activation='relu')(X4)
    X6 = Dense(4, activation='relu')(X5)
    X7 = Dense(1, activation='relu')(X6)
    outputs = Dense(1, activation='sigmoid')(X7)
    
    model = Model(input=inputs, output=outputs)
    model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'], )
    model.summary()

    return model

X_Train, Y_Train, X_Valid, Y_Valid, X_Test, Y_Test = load_dataset()

model = model()
model.fit(X_Train, Y_Train, epochs=12, batch_size=300, validation_data=(X_Valid,Y_Valid))

Y_Pred = model.predict(X_Test, batch_size=300)
Y_Pred = Y_Pred>0.5
test_acc = (1.0*np.sum(Y_Pred==Y_Test))/(1.0*Y_Test.shape[0])
print('Test accuracy = '+str(test_acc)+' on '+str(Y_Test.shape[0])+' exxamples.')

id = random.randint(1,100000000)
name = 'my_model'+str(id)+'.h5'
print('model saved as '+name)
model.save('models/'+name)