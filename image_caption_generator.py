
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image
import matplotlib.pyplot as plt

training_image_path = 'archive/archive-8/Images'
testing_image_path = 'testing/testing/Images'

# storing the features extracted for each image from the CNN
training_data = {}

# storing features of testing images
testing_data = {}

training_directory = os.listdir(training_image_path)
testing_directory = os.listdir(testing_image_path)

vgg = VGG16()

# removing last 2 Linear layers of the VGG model to stop at feature extraction
vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

# function to extract and save features for use in LSTM
# load image, convert to array
# run image through CNN to get features
def extract_save_features(list, directory, image_path):
    for file in directory:
        path = os.path.join(image_path, file)
        image = load_img(path, target_size=(224,224))
        image_array = img_to_array(image)
        image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], image_array.shape[2])
        processed_image = preprocess_input(image_array)
        feature_extraction = vgg.predict(processed_image, verbose=0)
        list[file] = feature_extraction
    return list

training_data = extract_save_features(training_data, training_directory, training_image_path)

testing_directory = os.listdir(testing_image_path)
testing_data = extract_save_features(testing_data, testing_directory, testing_image_path)

# processing captions - stored in captions.txt
true_caption = {}
total_captions = []



max_caption = 0

with open('archive/archive-8/captions.txt', 'r') as captions:
    next(captions)
    captions_text = captions.read()

captions_text = captions_text.split('\n')

for line in captions_text:
    # caption format:
    # image name, caption
    # splitting name and caption
    split = line.split(',')

    if len(line) < 2:
        continue

    name = split[0]
    full_caption = split[1:]

    full_caption = ' '.join(full_caption)

    # clean captions
    full_caption = full_caption.lower()
    full_caption = full_caption.replace('[^A-Za-z]', '')
    full_caption = full_caption.replace('\s+', ' ')
    full_caption = 'startseq ' + full_caption + ' endseq'

    if name not in true_caption:
        true_caption[name] = []
        true_caption[name].append(full_caption)
    if name in true_caption:
        true_caption[name].append(full_caption)
    total_captions.append(full_caption)
    max_caption = max(len(full_caption.split()), max_caption)

len(total_captions)

len(true_caption)

# splitting into tokens

token = Tokenizer()
token.fit_on_texts(total_captions)
token_size = len(token.word_index) + 1
print(token_size)

print(max_caption)

# splitting the data into training and testing data
# 90 percent training and 10 percent testing
image_names = list(true_caption.keys())
split = int(len(image_names) * 0.90)
train = image_names[:split]
test = image_names[split:]

def generate_data(names, true_caption, extracted_feature, tokenizer, max, size, batch_size):
    data1 = list()
    data2 = list()
    target = list()   
    n = 0
    while 1:
        for name in names:
            n += 1
            captions = true_caption[name]

            for caption in captions:

                sequence = tokenizer.texts_to_sequences([caption])[0]
                
                for i in range(1, len(sequence)):
                    input = sequence[:i]
                    output = sequence[i]
                    
                    input = pad_sequences([input], maxlen=max)[0]

                    output = to_categorical([output], num_classes=size)[0]
                    
                    data1.append(extracted_feature[name][0])
                    data2.append(input)
                    target.append(output)

            if n == batch_size:
                data1 = np.array(data1)
                data2 = np.array(data2)
                target = np.array(target)
                yield [data1, data2], target
                data1 = list()
                data2 = list()
                target = list() 
                n = 0

# creating the model 

inputs = Input(shape=(4096,))
dropout = Dropout(0.4)(inputs)
linear = Dense(256, activation='relu')(dropout)

inputs2 = Input(shape=(max_caption,))
embedding = Embedding(token_size, 256, mask_zero=True)(inputs2)
dropout2 = Dropout(0.4)(embedding)
lstm = LSTM(256)(dropout2)

# decoder model
decoder1 = add([linear, lstm])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(token_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# training 
epochs = 20
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    print(i, ' of ', epochs)
    generator = generate_data(train, true_caption, training_data, token, max_caption, token_size, batch_size)
    
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# converting the numbers back to words
def convert_number(int, tokenizer):
    for word, i in tokenizer.word_index.items():
        if i == int:
            return word
    return None

# predict caption for an image
def predict_caption(model, image, tokenizer, max_caption):
    
    in_text = 'startseq'
   
    for i in range(max_caption):
       
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        
        sequence = pad_sequences([sequence], max_caption)
        
        predicted = model.predict([image, sequence], verbose=0)
        
        predicted = np.argmax(predicted)
        
        word = convert_number(predicted, tokenizer)
        
        if word is None:
            break
        
        in_text += " " + word
        
        if word == 'endseq':
            break
      
    return in_text

# calcluate accuracy on test data
actual_list = []
predicted_list = []
i = 0
for key in test:
    print(i, ' of ', len(test))

    caption = true_caption[key] 
    print(type(caption)) 
    predicted_caption = predict_caption(model, training_data[key], token, max_caption)

    actual = [c.split() for c in caption]

    predicted_caption = predicted_caption.split()

    # append to the list
    actual_list.append(actual)
    predicted_list.append(predicted_caption)
    i = i + 1
    
# calcuate accuracy with corpus bleu measurement
print(corpus_bleu(actual_list, predicted_list))

def generate_caption(image_name, training):
    if training:
        img_path = os.path.join(training_image_path, image_name)
    else:
        img_path = os.path.join(testing_image_path, image_name)
    
    image = Image.open(img_path)
    if training:
        captions = true_caption[image_name] 
        print('true captions:')  
        for caption in captions:
            print(caption.replace('startseq','').replace('endseq',''))
    if training:
        prediction = predict_caption(model, training_data[image_name], token, max_caption)
    else:
        prediction = predict_caption(model, testing_data[image_name], token, max_caption)
    print('Predicted Caption:')
    print(prediction.replace('startseq','').replace('endseq',''))
    plt.imshow(image)

generate_caption("101654506_8eb26cfb60.jpg", True)

generate_caption(test[10], True)

generate_caption(test[18], True)

generate_caption(test[28], True)

generate_caption(test[100], True)

generate_caption("107_107.jpg", False)