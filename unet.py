
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

# Пути к данным
IMAGE_PATH = "./camvid/images/"  # Путь к изображениям
MASK_PATH = "./camvid/masks/"    # Путь к маскам

# Параметры модели
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 32  # Количество классов в CamVid


def load_data(image_path, mask_path):
    images_list = []
    masks_list = []
    
    for img_name in os.listdir(image_path):
        # Загружаем изображения
        img = load_img(os.path.join(image_path, img_name), target_size=(IMG_HEIGHT, IMG_WIDTH))
        img = img_to_array(img) / 255.0  # Нормализация
        images_list.append(img)
        
        # Загружаем маски
        mask_name = img_name.replace(".png", "_L.png")  # Предполагается определенный формат
        mask = load_img(os.path.join(mask_path, mask_name), target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
        mask = img_to_array(mask).astype(np.int32)
        masks_list.append(mask)
    
    return np.array(images_list), np.array(masks_list)


# Загрузка данных
images, masks = load_data(IMAGE_PATH, MASK_PATH)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# One-hot кодирование масок
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)


def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    inputs = Input(input_size)
    
    # Энкодер
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Боттлнек
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
    
    # Декодер
    up1 = UpSampling2D(size=(2, 2))(conv3)
    concat1 = Concatenate()([up1, conv2])
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(concat1)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv4)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    concat2 = Concatenate()([up2, conv1])
    conv5 = Conv2D(64, (3, 3), activation="relu", padding="same")(concat2)
    conv5 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv5)
    
    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(conv5)

    model = Model(inputs, outputs)
    return model


# Создание модели
model = unet_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Настройка обучения
BATCH_SIZE = 16
EPOCHS = 20

# Обучение
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# Сохранение модели
model.save("unet_camvid.h5")


# Функция для отображения
def visualize_prediction(image, mask, prediction):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image)
    
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(np.argmax(mask, axis=-1))
    
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(np.argmax(prediction, axis=-1))
    plt.show()


# Пример предсказания
sample_image = X_test[0]
sample_mask = y_test[0]
predicted_mask = model.predict(np.expand_dims(sample_image, axis=0))

visualize_prediction(sample_image, sample_mask, predicted_mask[0])
