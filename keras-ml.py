import sys

for p in sys.path:
    print(p)

from keras.models import Sequential
from keras.layers import Dense, Dropout

from data import ppls

###################
# Step 1 - Set-up
###################

# Load all the data
x_train, y_train = ppls.load_data()

model = Sequential()
model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

h = model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

print(h)