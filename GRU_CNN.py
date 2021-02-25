# Importing all the necessary libraries
import csv
import keras
from keras import optimizers
from Multimodal_baseline_Functions import *
from keras.utils.vis_utils import plot_model
import os
from keras.layers import Flatten, Bidirectional
from keras.layers.recurrent import GRU

# Storing directory of embeddings
Fast_DIR = "E:\Meme"

EMBEDDING_DIM = 300

# Assigning class weights
class_weight = {1: 1.1,
                0: 1.}

# Defining model with Adam optimizer
adam = keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = optimizers.SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
adadelta = optimizers.Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.0)


def Image_model(base_model):
    # Freezing all the trainable layers
    for layer in base_model.layers:
        layer.trainable = False

    # Creating output layer
    x = base_model.output
    # Adding pooling layer before the output
    x = GlobalAveragePooling2D()(x)
    return x


def read_data(file_name):
    # Opening file
    with open(file_name, 'r', encoding="utf8") as f:
        # Creating empty set and dictonary for vocab and word respectively
        word_vocab = set()
        word2vector = {}
        # Iterating over each line of file
        for line in f:
            # Spliting lines
            line_ = line.strip()
            # Splitting words
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:], dtype=float)
    print("Total Words in DataSet:", len(word_vocab))
    return word_vocab, word2vector


# Dividing data in test, train, validation

Training_path = r'E:\Meme\data\train_1.csv'
Validation_path = r'E:\Meme\data\val_1.csv'
Testing_path = r'E:\Meme\data\test.csv'

training_DF, testing_DF, validation_DF = preprocess_text(Training_path, Validation_path, Testing_path)

# Processing image and text for each set
img_dir = r'E:\Meme\data\train_img'
test_img_dir = r'E:\Meme\data\test_img'
train_img_path = create_img_path(training_DF, 'image_name', img_dir)
test_img_path = create_img_path(testing_DF, 'image_name', test_img_dir)
val_img_path = create_img_path(validation_DF, 'image_name', img_dir)

# Processing the text
training_DF['sentence'] = training_DF['sentence'].apply(clean_text)
testing_DF['sentence'] = testing_DF['sentence'].apply(clean_text)
validation_DF['sentence'] = validation_DF['sentence'].apply(clean_text)

# Vectorising text
# process the whole observation into single list
train_text_list = list(training_DF['sentence'])
test_text_list = list(testing_DF['sentence'])
val_text_list = list(validation_DF['sentence'])

# Creating vectors for train, test, validation
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(train_text_list)
sequences_train = tokenizer.texts_to_sequences(train_text_list)
sequences_test = tokenizer.texts_to_sequences(test_text_list)
sequences_val = tokenizer.texts_to_sequences(val_text_list)

x_train = preprocessing.sequence.pad_sequences(sequences_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(sequences_test, maxlen=maxlen)
x_val = preprocessing.sequence.pad_sequences(sequences_val, maxlen=maxlen)

# encoding all the labels
y_train = training_DF['label']
y_val = validation_DF['label']

y_test = ''
# Creating train, test, val, generator for meme
img_txt_gen_train = img_text_generator(train_img_path, x_train, y_train, batch_size=32)
img_txt_gen_test = img_text_test_generator(test_img_path, x_test, batch_size=1)
img_txt_gen_val = img_text_generator(val_img_path, x_val, y_val, batch_size=1)

# Creating train, test, val, generator for text
txt_gen_train = text_generator(x_train, y_train, batch_size=32)
txt_gen_test = text_generator(x_test, y_test, batch_size=1)
txt_gen_val = text_generator(x_val, y_val, batch_size=1)

# Creating train, test, val, generator for image
img_gen_train = image_generator(train_img_path, training_DF, batch_size=32)
img_gen_test = image_generator(test_img_path, testing_DF, batch_size=1)
img_gen_val = image_generator(val_img_path, validation_DF, batch_size=1)

# Creating vocabulary with  embeddings
vocab, w2v = read_data(os.path.join(Fast_DIR, "cc.ta.300.vec"))

# Creating word index
word_index = tokenizer.word_index
num_tokens = len(word_index)

# Creating embeddding weight matrix
embedding_matrix = np.zeros((num_tokens + 1, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector = w2v.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Creating embedded layer using embedded matrix as weight matrix
embedding_layer = Embedding(num_tokens + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)

# Defining input layer
main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')

# Defining embedding layer which will encode the input sequence
embedded_sequences = embedding_layer(main_input)

GRU1 = GRU(32, return_state=True)
encoder_outputs, state_h, state_c = Bidirectional(GRU1)(embedded_sequences)
states = [state_h, state_c]

# Defining second GRU
GRU2 = GRU(32, return_sequences=True, return_state=True)
# Adding initial state as state of  previous GRU
decoder_out, _, _ = Bidirectional(GRU2)(embedded_sequences, initial_state=states)
# Adding dense layer with 500 neurons
decoder_dense = (Dense(500, activation='relu'))
# GRU output
GRU_out = (decoder_out)
# Flattening
GRU_out = Flatten()(GRU_out)
# Output of text model
txt_out = Dense(1, activation='softmax')(GRU_out)

# Defining text model
txt_model = Model(inputs=[main_input], outputs=txt_out)

# compiling text model
txt_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

# Training text model
txt_model.fit_generator(txt_gen_train, epochs=10, validation_steps=149, steps_per_epoch=2, validation_data=txt_gen_val,
                        shuffle=False, class_weight=class_weight)

# Saving text model
txt_model.save('GRU_txt_model.h5')

steps = len(y_val.values)

# Predicting labels of the test set
y_pred_txt = (txt_model.predict_generator(txt_gen_val, steps=steps))
y_pred_txt = np.round(list(itertools.chain(*y_pred_txt)))
# Defining true labels
y_true = y_val.values


labels = [1, 0]
print(y_true.shape)
print(y_pred_txt.shape)

img_input = layers.Input(shape=(300, 300, 3))
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
output = layers.Dense(1, activation='softmax')(x)
CNN_model = Model(img_input, output)

# create base model with imagenet weights

# Building img_prediction layer using Image_model
base_img = Image_model(CNN_model)
img_prediction_layer = Dense(1, activation='softmax')(base_img)

# Defining image model
img_model = Model(inputs=[CNN_model.input], outputs=img_prediction_layer)

# Compiling image model
img_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

# Plotting image model
plot_model(img_model, to_file='CNN_img_model.png', show_shapes=True, show_layer_names=True)

# Training image model
img_model.fit_generator(img_gen_train, epochs=10, validation_steps=149, steps_per_epoch=2, validation_data=img_gen_val,
                        shuffle=False, class_weight=class_weight)

# Saving image model
img_model.save('CNN_img_model.h5')

steps = len(y_val.values)

# Predicting labels using image model
y_pred_img = (img_model.predict_generator(img_gen_val, steps=steps))
y_pred_img = np.round(list(itertools.chain(*y_pred_img)))
y_true = y_val.values

# Confusion matrix for image classifier
labels = [1, 0]

txt_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
img_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

# Concatenating output of both classifiers
con_layer = keras.layers.concatenate([txt_model.output, img_model.output])
out = Dense(1, activation='softmax')(con_layer)

# Defining model input and output
com_model = Model(inputs=[img_model.input, txt_model.input], outputs=out)

# Using Stochastic gradient descent with optimizer
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
com_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

# Training model
com_model.fit_generator(img_txt_gen_train, epochs=10, validation_steps=149, steps_per_epoch=2,
                        validation_data=img_txt_gen_val, shuffle=False, class_weight=class_weight)

# Saving combined model
com_model.save('GRU_CNN_mul_model.h5')
print("Saving combined model")

y_pred_com = (com_model.predict_generator(img_txt_gen_test, steps=steps))
y_pred_com = np.round(list(itertools.chain(*y_pred_com)))


def write_result(filename, ids, predictions):
    with open(filename, 'w', encoding='utf-8') as w:
        writer = csv.writer(w, delimiter=",")
        writer.writerow(['id', 'label'])
        for id, pred in zip(ids, predictions):
            writer.writerow([id, int(pred)])


steps = len(test_img_path)
y_pred_test_com = (com_model.predict_generator(img_txt_gen_test, steps=steps))
y_pred_test_com = np.round(list(itertools.chain(*y_pred_test_com)))
print(y_pred_test_com[:15])
ids = testing_DF['image_name']
write_result('./result.csv', ids, y_pred_test_com)

print("Finish predict combined model")
