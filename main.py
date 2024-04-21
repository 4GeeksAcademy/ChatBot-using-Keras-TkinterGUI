from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import random
import json
import numpy as np

# Descargar recursos de NLTK (ejecutar solo una vez)
nltk.download('punkt')
nltk.download('wordnet')

# Cargar datos de intents.json
with open('intents.json') as file:
    intents = json.load(file)

# Preprocesamiento de datos
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizar y lematizar palabras
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        # Agregar a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizar y eliminar duplicados
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))

# Crear datos de entrada y salida
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Convertir a numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

# Definición del modelo
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compilación del modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Guardar el modelo
model.save('chatbot_model.h5')
print("Modelo guardado como 'chatbot_model.h5'")