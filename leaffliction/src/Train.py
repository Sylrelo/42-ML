
import argparse
import datetime
import json
import os
import cv2
import tensorflow as tf

from keras import Sequential, layers, callbacks, optimizers, losses
from Transformation import transform_with_mask
from matplotlib import pyplot as plt
from Augmentation import balance


def _build_model(img_height, img_width, classes):
    _classes_count = len(classes)

    _model = Sequential(
        [
          # Couche d'entrée, doit correspondre à la taille de l'image et du
          # nombre de composants de couleurs (3, RGB)
          layers.Input(shape=(img_height, img_width, 3)),

          # Couche convolutionnelle.
          # 16 représente de nombre de neurones.
          # ReLU (Rectified Linear Unit) est utilisé pour introduire de la
          #    non-linéarité dans le modèle
          layers.Conv2D(16, (3, 3), activation="relu"),
          # Couche de Pooling MAX
          # Réduit la dimensionnalité des données, le temps de traitement,
          #   et limite l'overfitting
          layers.MaxPooling2D(),

          layers.Conv2D(32, (3, 3), activation="relu"),
          layers.MaxPooling2D(),

          layers.Conv2D(64, (3, 3), activation="relu"),
          layers.MaxPooling2D(),

          layers.Conv2D(64, (3, 3), activation="relu"),
          layers.MaxPooling2D(),

          layers.Conv2D(128, (3, 3), activation="relu"),
          layers.MaxPooling2D(),

          # Transforme les données en un vecteur unidimensionnel, nécessaire
          #   pour l'utilisation d'une couche dense
          layers.Flatten(),

          # Couche entièrement connecté à la couche précendente
          layers.Dense(128, activation="relu"),

          # Désactivation aléatoire de XX% des neurones pendant le train.
          # Réduit l'overfitting
          layers.Dropout(0.35),

          # Couche de sortie, le nombre de neuronnes doit être égal au nombre
          #   de classe à prédire.
          # softmax : Normalise les résultats en probabilité, la somme des
          #   sorties sera égale à 1.
          # Exemple pour 3 classes ["patate", "pomme", "pêche"]
          #                        [0.2,      0.6,      0,2] = 1.0
          # La prédiction sera "pomme".
          layers.Dense(_classes_count, activation="softmax"),
        ]
    )

    # Adam (Adaptive Moment Estimation)
    # Permet d'adapter le taux d'apprentissage au fil du temps
    optimizer = optimizers.Adam(
        learning_rate=0.001
    )

    loss = losses.SparseCategoricalCrossentropy()

    _model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    return _model


def _run_model(
        model: Sequential,
        train_dataset,
        validation_dataset,
        save_checkpoints=None
):
    _cb = []
    # Early Stopping
    # Permet d'arrêter l'entrainement du modèle pour éviter le
    #   sur-entrainement en surveillant l'évolution de la perte
    stop_early = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        start_from_epoch=5
    )
    _cb.append(stop_early)

    if save_checkpoints is True:
        date = f"{datetime.datetime.now().strftime('%Y%m%d')}"
        filepath = './weights/' + date + \
            '/{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.weights.h5'

        checkpoint = callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,
            monitor='val_loss'
        )
        _cb.append(checkpoint)

    model.summary()

    return model.fit(
      train_dataset,
      validation_data=validation_dataset,
      epochs=50,
      callbacks=_cb,
    )


def _create_missing_directory(path: str):
    if os.path.exists(path) is False:
        os.makedirs(path)


def _save_dataset(classnames, dest_path, dataset):
    if os.path.exists(dest_path) and os.path.isfile(dest_path):
        print("Dataset destination is not a directory.")
        exit(1)
    elif os.path.exists(dest_path) is False:
        os.makedirs(dest_path)

    print("Saving dataset...")
    for i, (images, labels) in enumerate(dataset):
        for j, (image, label) in enumerate(zip(images, labels)):
            class_name = classnames[label]
            filename = f'image_{i * 32 + j}.jpg'
            dst = os.path.join(dest_path, class_name)
            save_path = os.path.join(dst, filename)
            _create_missing_directory(dst)
            tf.keras.preprocessing.image.save_img(save_path, image)


def _transform_images(directory_path: str):
    print("==== Transforming Images ====")
    for root, _, files in os.walk(directory_path):
        if len(files) == 0:
            continue

        for file in files:
            if not file.lower().endswith(".jpg"):
                continue
            filepath = os.path.join(root, file)
            print(filepath)
            transformed = transform_with_mask(filepath)
            cv2.imwrite(filepath, transformed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "src",
        help="Source directory containing images to train \
          (will be overwritten with transformations)"
    )

    parser.add_argument(
        "--dataset-dest",
        help="Dataset saving directory",
        required=True
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Agument the image before training."
    )

    parser.add_argument(
        "--transform",
        action="store_true",
        help="Transform the image before training."
    )

    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save model at each epochs."
    )

    parser.add_argument(
         "--batch-size",
         type=int,
    )

    parser.add_argument(
         "--random-seed",
         type=int,
    )

    parser.add_argument(
         "--split",
         type=float,
    )

    args = parser.parse_args()

    assert os.path.isdir(args.src), "Source is not a directory."
    assert os.path.exists(args.src), "Source does not exists."

    dir_path = args.src

    validation_split = max(0.2, min(0.8, args.split or 0.3))
    random_seed = max(0, args.random_seed or 42)
    img_height = 255
    img_width = 255
    batch_size = max(1, min(128, args.batch_size or 32))

    print(f"Batch Size: {batch_size}")
    print(f"Random Seed: {random_seed}")
    print(f"Validation Split: {validation_split}")

    if args.augment is True:
        balance(args.src, args.src)

    if args.transform is True:
        _transform_images(dir_path)

    train_data = tf.keras.utils.image_dataset_from_directory(
      dir_path,
      validation_split=validation_split,
      subset="training",
      seed=random_seed,
      image_size=(img_height, img_width),
      batch_size=batch_size,
      shuffle=True,
    )

    class_names = train_data.class_names

    validation_data = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        validation_split=validation_split,
        subset="validation",
        seed=random_seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
    )

    train_data_path = os.path.join(args.dataset_dest, "train")
    validation_data_path = os.path.join(args.dataset_dest, "validation")
    _save_dataset(class_names, train_data_path, train_data)
    _save_dataset(class_names, validation_data_path, validation_data)

    train_data = train_data \
        .cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_data = validation_data \
        .cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    normalize = layers.Rescaling(1.0 / 255.0)
    train_data = train_data.map(lambda x, y: (normalize(x), y))
    validation_data = validation_data.map(lambda x, y: (normalize(x), y))

    model = _build_model(
        img_height=img_height,
        img_width=img_width,
        classes=class_names
    )

    history = _run_model(
        model=model,
        train_dataset=train_data,
        validation_dataset=validation_data,
        save_checkpoints=args.save_checkpoints
    )

    model.save("lopez-4.tfmodel.h5")
    with open('./lopez-4.classnames.json', 'w') as f:
        json.dump(class_names, f)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')

    plt.show()
