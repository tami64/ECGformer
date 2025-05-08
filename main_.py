import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,ConfusionMatrixDisplay,classification_report

EPOCHE = 10
BATCH_SIZE = 64

def readucr(filename):
    data = pd.read_csv(filename,header=None)
    nRow, nCol = data.shape
    print(f'There are {nRow} train rows and {nCol} columns')
    y = data.iloc[:,-1].astype(int).to_numpy()
    x = data.iloc[:, :-1]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(x)
    x = pd.DataFrame(standardized_data, columns=x.columns).to_numpy()
    return x, y.astype(int)

x_train, y_train = readucr('mitbih_train.csv')
x_test, y_test = readucr('mitbih_test.csv')

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

idx = np.random.permutation(len(x_test))
x_test = x_test[idx]
y_test = y_test[idx]


n_classes = len(np.unique(y_train))


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=16,
    num_heads=8,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128,64],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)
model.summary()

checkpoint_filepath = './tmp/checkpoint'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHE,
    batch_size=BATCH_SIZE,
    callbacks=[model_checkpoint_callback],
    validation_data=(x_test, y_test)
)
model.load_weights(checkpoint_filepath)

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

y_pred_probs = model.predict(x_test)
y_pred = y_pred_probs.argmax(axis=-1)

cm = confusion_matrix(y_test, y_pred,normalize = 'pred')

test_accuracy = accuracy_score(y_test, y_pred)

train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
'''
indVal_accuracy = np.array(history.history['val_accuracy']).argmax()
BestTrainLoss = train_loss[indVal_accuracy]
BestTrainAcc = train_accuracy[indVal_accuracy]

print("Train Loss", train_loss)
print("Test Loss:", loss)
print("Test Accuracy:", test_accuracy)
'''
report = classification_report(y_test, y_pred,target_names=['N','S', 'V','F','Q'])
print(report)
with open('report.txt', 'a') as f:
    f.write(report)
'''
with open('report.txt', 'a') as f:
    f.write("\n" + "Best Test Loss:" + str(loss) + "\n")
    f.write("Best Test Accuracy:" + str(test_accuracy) + "\n")
    f.write("Best Train Loss:" + str(BestTrainLoss) + "\n")
    f.write("Best Train Accuracy:" + str(BestTrainLoss))
'''

plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
#plt.plot()
plt.savefig('loss_plot.jpg', format='jpg')

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.plot()
plt.savefig('accuracy_plot.jpg', format='jpg')

Classes_caption = ['N','S', 'V','F','Q']
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Classes_caption)
fig, ax = plt.subplots(figsize=(6,4))
disp.plot(ax=ax)
plt.savefig('confusion_matrix.jpg', format='jpg')