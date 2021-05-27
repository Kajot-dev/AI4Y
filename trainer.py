import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import pandas as pd
import cv2
if (tf.test.is_gpu_available()):
    print("TF using GPU!")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("Allow growth enabled on", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
labels = {
  "age": {
    "0": "0-9",
    "1": "10-19",
    "2": "20-29",
    "3": "30-49",
    "4": "50-69",
    "5": "70+"
  },
  "gender": {
    "0": "Male",
    "1": "Female"
  },
  "race": {
    "0": "White",
    "1": "Black",
    "2": "Indian",
    "3": "Asian",
    "4": "Latino_Hispanic"
  }
}

batch_size = 64
def GetSet(path):
    training_set = pd.read_csv(path, ";")
    training_imgs = ["{}.jpg".format(x) for x in list(training_set.file)]
    training_labels_1 = list(training_set['age'])
    training_labels_2 = list(training_set['gender'])
    training_labels_3 = list(training_set['race'])
    training_set = pd.DataFrame(
        {'images': training_imgs, 'age': training_labels_1, 'gender': training_labels_2, 'race': training_labels_3})
    training_set.age = training_set.age.astype(str)
    training_set.gender = training_set.gender.astype(str)
    training_set.race = training_set.race.astype(str)
    training_set['merged_class'] = training_set['age'] + training_set['gender'] + training_set['race']
    length = len(training_set["images"])
    return training_set, length
def PrepareValGen(dataframe, dir=""):
    generator = ImageDataGenerator(rescale=1. / 255,)
    generator = generator.flow_from_dataframe(
        dataframe=dataframe,
        directory=dir,
        x_col="images",
        y_col="merged_class",
        class_mode="categorical",
        target_size=(224, 224),
        batch_size=batch_size
    )
    return generator
def PrepareTrainGen(dataframe, dir=""):
    generator = ImageDataGenerator(rescale=1./255,
        rotation_range=30,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode="nearest")
    generator = generator.flow_from_dataframe(
        dataframe=dataframe,
        directory=dir,
        x_col="images",
        y_col="merged_class",
        class_mode="categorical",
        target_size=(224, 224),
        batch_size=batch_size
    )
    classes = generator.class_indices
    return generator, classes

def GetGenerators():
    training_set, train_val = GetSet("Dataset_mod/train_labeled.csv")
    test_set, test_val = GetSet("Dataset_mod/test_labeled.csv")
    train_generator, classes = PrepareTrainGen(training_set, "Dataset_mod/")
    test_generator = PrepareValGen(test_set, "Dataset_mod/")
    SaveClassIndics(classes, True)
    return (train_generator, test_generator), (train_val, test_val)

def BuildTheCNN():
    model = Sequential([
        Conv2D(32, 3, padding='same', activation='relu',
               input_shape=(224, 224, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.25),
        Dense(60, activation='softmax')  # 60 classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy', 'accuracy'])
    model.summary()
    return model

def GetLabels(merged_class):
    classes = list(merged_class)
    classes_l = []
    classes_l.append(labels["age"][classes[0]])
    classes_l.append(labels["gender"][classes[1]])
    classes_l.append(labels["race"][classes[2]])
    return (classes, classes_l)

def SaveClassIndics(classes, isinv=False):
    if (isinv):
        classes = dict(map(reversed, classes.items()))
    with open('Dataset_mod/classes_indics.json', 'w+') as file:
        json.dump(classes, file)

def LoadClassIndics():
    with open('Dataset_mod/classes_indics.json', 'r') as fp:
        data = json.load(fp)
    return data
class CacheCallback(Callback):
    def on_epoch_end(self):
        tf.keras.backend.clear_session()
def SetUpCallback(epoch_steps=10):
    checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
    callback = ModelCheckpoint(filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_freq=int(epoch_steps/8)*batch_size) #save every epoch
    return callback, checkpoint_path

def TrainModel(model=None, generators = None, values=None, epochs=15, checkpointcallback=None , checkpoints=None, resume=0):
    if (model == None):
        model = BuildTheCNN()
    if (generators == None or values == None):
        generators, values = GetGenerators()
    if (checkpointcallback == None):
        checkpointcallback = SetUpCallback(values[0])
    if (checkpoints is not None):
        model.load_weights(checkpoints)
        print("Weights loaded!")
    else:
        model.save_weights(checkpointcallback[1].format(epoch=0))
    tf.keras.backend.clear_session()
    model.fit_generator(
        generators[0],
        steps_per_epoch=int(values[0]/8),
        epochs=epochs,
        validation_data=generators[1],
        validation_steps=int(values[1]/8),
        shuffle=True,
        callbacks=[checkpointcallback[0], CacheCallback()],
        verbose=1,
        initial_epoch=resume
    )
    tf.keras.backend.clear_session()
    SaveModel(model, "trained_fresh.h5")
    return model

def GetCheckpoints(checkpoint_fit="training_checkpoints/"):
    latest = tf.train.latest_checkpoint(checkpoint_fit)
    return latest

def ResumeTrainig(epoch_start, epochs=15):
    checkpoints = GetCheckpoints()
    print(checkpoints)
    TrainModel(resume=epoch_start, checkpoints=checkpoints, epochs=epochs)
def SaveModel(model, name="alpha_model.h5"):
    model.save(name)
def SaveModelFromWeights():
    model = BuildTheCNN()
    ckp = GetCheckpoints()
    print("Checkpoints ok!")
    model.load_weights(ckp)
    SaveModel(model)
def TrainFromModel(model, resume=0, epochs=15):
    new_model = TrainModel(model=model, resume=resume, epochs=epochs)
    SaveModel(new_model)
def ConvertToLabels(pred):
    pred = "".split(pred)
    final = [labels["age"][pred[0]], labels["gender"][pred[1]], labels["race"][pred[2]]]
    return final
def Predict(model, input_image=None, path=None):
    if (input_image is not None):
        if (input_image.shape != (224, 224, 3)):
            img = cv2.resize(input_image, (224, 224))
        img = img.reshape((1,224,224,3))
        img = tf.image.convert_image_dtype(img, tf.float32)
        input_image = img
    elif (path is not None):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.reshape((1,224,224,3))
        img = tf.image.convert_image_dtype(img, tf.float32)
        input_image = img
    predictions = model.predict_classes(input_image)
    class_indics = LoadClassIndics()
    prediction = class_indics[str(predictions[0])]
    final_predictions, l = GetLabels(prediction)
    return final_predictions, l
if (__name__ == "__main__"):
    TrainModel()