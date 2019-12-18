import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras import backend as K

BATCH_SIZE = 1

# 
class MyGD(Optimizer):

    def __init__(self, lr=0.01):
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64',name = 'iterations')
            self.lr = K.variable(lr,name='lr')
            #self.step_per_update = step_per_update
    
    def get_updates(self,loss,params):
        grads = self.get_gradients(loss, params) # get gradient
        self.updates = [K.update_add(self.iterations, 1)]
        self.weights = [self.iterations] 
        for p, g in zip(params, grads):
            # GD with fixed step_length lr
            new_p = p - self.lr * g
            # add constraint if any
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

my_gd = MyGD()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

fig = plt.figure()

for i in range(10):
    plt.subplot(2, 5, i+1)
    x_y = X_train[y_train == i]
    plt.imshow(x_y[0], cmap='gray', interpolation='none')
    plt.title("Class %d" % (i))
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()

print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('X_test.shape', X_test.shape)
print('y_test.shape', y_test.shape)

# reshaping the inputs
X_train = X_train.reshape(60000, 28*28)
X_test = X_test.reshape(10000, 28*28)

#print(X_train[0])

# normalizing the inputs
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

#print(X_train[0])

print('X_train.shape', X_train.shape)
print('X_test.shape', X_test.shape)

# 10 classes
y_train_kind = to_categorical(y_train, 10)
y_test_kind = to_categorical(y_test, 10)
print('y_train_kind shape:', y_train_kind.shape)
print('y_test_kind shape:', y_test_kind.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
# Add layer
model.add(Dense(10, input_dim=28*28, activation='softmax'))

# prints a summary representation of the model
model.summary()

# compiling the sequential model
model.compile(optimizer = my_gd, loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# training the model and saving metrics in history
history = model.fit(X_train, y_train_kind,
                    batch_size=BATCH_SIZE, epochs=50,
                    verbose=2,
                    validation_data=(X_test, y_test_kind))

# plotting the metrics
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()


# evaluate model on test data
[test_loss, test_acc] = model.evaluate(X_test, y_test_kind) 
print("Evaluation result on Test Data:\nLoss = {}\nAccuracy = {}".format(test_loss, test_acc))