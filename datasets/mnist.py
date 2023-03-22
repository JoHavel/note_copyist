from keras.datasets import mnist

shape = (28, 28, 1)

train_len = 60000
val_len = 10000
test_len = 10000

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_val = X_train[train_len:]
X_train = X_train[:train_len]

y_val = y_train[train_len:]
y_train = y_train[:train_len]

X_train = X_train.reshape(X_train.shape[0], *shape).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], *shape).astype('float32') / 255.0
