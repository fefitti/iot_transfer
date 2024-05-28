import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definição do modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Compilação do modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Pré-processamento e aumento de dados
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'train/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# Treinamento do modelo
model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1)

# Aplicação de Transfer Learning
base_model = tf.keras.applications.VGG16(input_shape=(150, 150, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

# Adicionando camadas adicionais ao modelo base
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

# Combinando o modelo base com as camadas adicionais
model_transfer = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

# Compilação do modelo transferido
model_transfer.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Treinamento do modelo transferido
model_transfer.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=5,
      verbose=1)
