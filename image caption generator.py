# importing the required libraries
from keras.utils import plot_model
import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

# to track the progress of loops
from tqdm import tqdm, notebook
tqdm().pandas()

# storing paths to required datasets
Flicker8k_Dataset = "/Users/Swa/Documents/CODING/Image Caption/Flickr_Data/Images"
Flicker8k_text = "/Users/Swa/Documents/CODING/Image Caption/Flicker8k_text"

# loading a text file into memory
def load_doc(filename):
    
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# get all images with their captions
def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')

    # a dictionary to map all images to their corresponding caption/s
    descriptions = {}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        
        # if image not in dictionary, add the image to the dictionary and begin a list of captions
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        
        # if image already in dictionary, add the caption to its list
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions


# data cleaning - lower casing, removing puntuations and words containing numbers
def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption.replace("-", " ")
            desc = img_caption.split()
            
            # convert every word to lowercase
            desc = [word.lower() for word in desc]
            
            # remove punctuations from each token
            desc = [word.translate(table) for word in desc]
            
            # remove hanging 's and 'a'
            desc = [word for word in desc if(len(word) > 1)]
            
            # remove tokens with numbers in them
            desc = [word for word in desc if(word.isalpha())]
            
            # convert back to string by joining all the words with space between them
            img_caption = ' '.join(desc)
            
            # replace the existing captions with processed caption
            captions[img][i] = img_caption
    return captions


# build vocabulary of all unique words
def text_vocabulary(descriptions):
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab


# save all descriptions in one file
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = "\n".join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()


# extract features from the image
def extract_features(directory):
    
    # here, we shall use the Xception model of image captioning with average pooling layers
    model = Xception(include_top=False, pooling='avg')
    
    # dictionary to store the features
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img
        
        # skipping unrequired files
        if filename == '/Users/Swa/Documents/CODING/Image Caption/Flickr_Data/.DS_Store':
            continue
        image = Image.open(filename)
        
        # processing image according the model requirements, like resizing, normalising etc.
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        
        # supplying the model with the processed image
        feature = model.predict(image)
        features[img] = feature
    return features


# load the data
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos


# loading clean_descriptions
def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words) < 1:
            continue
        
        # extracting images and their corresponding captions
        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            
            # adding 'start' and 'end' tokens to each description
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
    return descriptions


# loading all features
def load_features(photos):
    all_features = load(open("features.p", "rb"))
    
    # selecting only needed features
    features = {k: all_features[k] for k in photos}
    return features


# converting dictionary to clean list of descriptions
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# creating tokenizer class
# this will vectorise text corpus
# each integer will represent a token in the dictionary
def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer


# calculate maximum length of descriptions
def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)


# create input-output sequence pairs from the image description
# data generator, used by model.fit_generator()
def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            
            # retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(
                tokenizer, max_length, description_list, feature)
            yield [[input_image, input_sequence], output_word]



# helper function to create sequences
def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    
    # walk through each description for the image
    for desc in desc_list:
        
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            
            # pad input sequences to the max length description
            # this is done so that all descriptions are of same length
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            
            # store the calculated values
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


# define the captioning model
def define_model(vocab_size, max_length):
    
    # features from the CNN model squeezed from 2048 to 256 nodes, for the image
    inputs1 = Input(shape=(2048,))
    # adding a dropout layer for generalizing by removing half random nodes
    fe1 = Dropout(0.5)(inputs1)
    # add a dense layer in end to drill down to 256 nodes
    fe2 = Dense(256, activation='relu')(fe1)
    
    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    # pass through an embedding layer
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    # add drop out layer
    se2 = Dropout(0.5)(se1)
    # finally add an LSTM layer (a type of RNN)
    se3 = LSTM(256)(se2)
    
    # Merging both models
    decoder1 = add([fe2, se3])
    # pass this through a dense layer with relu activation
    decoder2 = Dense(256, activation='relu')(decoder1)
    # to get an output, we use the softmax function
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile the model, using cross entropy loss as loss function
    # used 'Adam' optimizer as it is a good general-purpose optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

if __name__ == "__main__":
  
  # Set the paths according to project folder in you system
  dataset_text = "/Users/Swa/Documents/CODING/Image Caption/Flicker8k_text"
  
  # we prepare our text data
  filename = dataset_text + "/" + "Flickr8k.token.txt"
  
  # loading the file that contains all data
  # mapping them into descriptions dictionary img to 5 captions
  descriptions = all_img_captions(filename)
  print("Length of descriptions =", len(descriptions))
  
  # cleaning the descriptions
  clean_descriptions = cleaning_text(descriptions)
  
  # building vocabulary
  vocabulary = text_vocabulary(clean_descriptions)
  print("Length of vocabulary = ", len(vocabulary))
  
  # saving each description to file
  save_descriptions(clean_descriptions, "descriptions.txt")
  
  # 2048 feature vector
  features = extract_features(Flicker8k_Dataset)
  dump(features, open("features.p", "wb"))
  features = load(open("features.p", "rb"))


  filename = Flicker8k_text + "/" + "Flickr_8k.trainImages.txt"
  train_imgs = load_photos(filename)
  train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
  train_features = load_features(train_imgs)

  # give each word an index, and store that into tokenizer.p pickle file
  tokenizer = create_tokenizer(train_descriptions)
  dump(tokenizer, open('tokenizer.p', 'wb'))
  vocab_size = len(tokenizer.word_index) + 1
  print(vocab_size)

  # determine the max length of a description
  max_length = max_length(descriptions)
  print(max_length)

  # You can check the shape of the input and output for your model
  [a, b], c = next(data_generator(train_descriptions,
                                  features, tokenizer, max_length))
  print(a.shape, b.shape, c.shape)

  # train our model
  print('Dataset: ', len(train_imgs))
  print('Descriptions: train=', len(train_descriptions))
  print('Photos: train=', len(train_features))
  print('Vocabulary Size:', vocab_size)
  print('Description Length: ', max_length)
  model = define_model(vocab_size, max_length)
  epochs = 10
  steps = len(train_descriptions)
  
  # making a directory models to save our models
  os.mkdir("models")
  for i in range(epochs):
      generator = data_generator(
          train_descriptions, train_features, tokenizer, max_length)
      model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
      model.save("models/model_" + str(i) + ".h5")