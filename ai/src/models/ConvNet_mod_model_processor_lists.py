# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from keras.callbacks import LearningRateScheduler,ModelCheckpoint,TensorBoard
from keras.optimizers import RMSprop,SGD,Adam
import keras.applications as modelList
from keras.applications import ResNet152, ResNet152V2,DenseNet169, ResNet50,resnet,VGG16,vgg16,resnet_v2
from keras.layers import Input, Dropout,Flatten,Dense,GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras.applications import imagenet_utils
#from imutils import paths
from random import shuffle
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import h5py
import cv2
from csv import reader


#Needed for top layers of model
from keras.layers import BatchNormalization
from keras import regularizers

#Adds top classification layer to transfer learning model
def top_layers(baseModel, classes, D):
	# initialize the head model that will be placed on top of
	# the base, then add a FC layer
	l2_constant=0.02
	first_dropout_rate=0.5
	dropout_rate=0.5
	headModel = baseModel.output
	headModel = GlobalAveragePooling2D()(headModel)
	headModel = Flatten()(headModel)
	headModel = Dense(D, activation="relu",kernel_regularizer=regularizers.l2(l2_constant))(headModel)

	# add a softmax layer
	headModel = Dense(classes, activation="softmax")(headModel)

	# return the model
	return headModel

#modifies learning rate based on epoch
def step_decay(epoch):
    initAlpha=.0020
    if epoch <= 20:
        alpha=initAlpha
    elif epoch <= 40:
        alpha=initAlpha*0.75
    elif epoch <= 60:
        alpha=initAlpha*0.50
    elif epoch <= 80:
        alpha=initAlpha*0.25
    elif epoch <=100:
        alpha=initAlpha*0.175
    elif epoch <= 125:
        alpha=initAlpha*0.10
    else:
        alpha=initAlpha*0.05
    return float(alpha)

    
#Loads in image data, labels from class subfolder
def create_data_arrays_list(imageData, image_dimension, verbose=-1):
    # initialize the list of features and labels
    data = []
    labels = []
    i=0

    # loop over the input images
    for image_input in imageData:
        # load the image
        images_dir = os.getcwd() + "/images/"
        image = cv2.imread(images_dir+image_input[0][:])
        
        alpha=1.3
        beta=0.0
        
        #image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


        # resize image, convert image to grayscale, then back to original color scheme for ResNet50, VGG16
        #Original interpolation was cv2.INTER_AREA
        image = cv2.cvtColor(cv2.cvtColor(cv2.resize(image, (image_dimension, image_dimension),
            interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        
        #convert image into array format using Keras function
        image=img_to_array(image)

        #add image data to array
        data.append(image)
        
        #Change label names so that class numbers will correspond to chronological sequence; class names changed back further on
        if image_input[:][1] == 'Kanaa': append_label='00'+'Kanaa'
        elif image_input[:][1] == 'Black_Mesa': append_label='01'+'Black_Mesa'
        elif image_input[:][1] == 'Sosi': append_label='02'+'Sosi'
        elif image_input[:][1] == 'Dogoszhi': append_label='03'+'Dogoszhi'
        elif image_input[:][1] == 'Flagstaff': append_label='04'+'Flagstaff'
        elif image_input[:][1] == 'Tusayan': append_label='05'+'Tusayan'
        else: append_label='06'+'Kayenta'
        
        #write label name
        labels.append(append_label)

        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1,
            len(imageData)))
        i=i+1

    # return image array data, labels
    return (np.array(data), np.array(labels))



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--set", required=True,
	help="integer index of sherd train/test set")
#Model number (to differentiate different runs
ap.add_argument("-r", "--run", required=True,
	help="run number")
ap.add_argument("-d","--dir",required=False, help="set directory")
args = vars(ap.parse_args())

#name of output model
set_number=str(args["set"])
run_number=str(args["run"])
os.chdir("../../image_data")
set_directory = os.getcwd()
images_dir = set_directory
CNN_model = "ConvNetBase"
image_dimension=384
model_id=CNN_model + "_"+set_number+"_"+run_number
print(model_id)
image_batch_size=8

# load the CNN model network, head FC layer sets are left
# off
baseModel=modelList.ConvNeXtBase(
    model_name="convnext_base",
    include_top=False,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=(384,384,3),
    pooling=None,
)

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier

Class_size=7 # Number of ceramic types
headModel = top_layers(baseModel, Class_size, 512)

# place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers:
	layer.trainable = False

print(CNN_model +  "model loaded")

#Define directories to use based on set number for loading data, saving data
train_dataset=set_directory + "/Set_" + set_number +"/train_" + set_number + ".csv"
test_dataset=set_directory + "/Set_" + set_number +"/test_" + set_number + ".csv"
models_dir = set_directory +"/Set_" + set_number +"/models/"

print(train_dataset)
print(test_dataset)



#Parameters for ImageDataGenerator
shift=0.0
zoom=0.3


# construct the image generator for data augmentation
#fill_mode is value put into empty spaces created by rotation or zooming; cval=1.0 means white
aug = ImageDataGenerator(rotation_range=180,
	horizontal_flip=False, vertical_flip=False, width_shift_range=shift, height_shift_range=shift, zoom_range=zoom, fill_mode="constant",cval=1.0)

# Get lists of train images, types from file
print("[INFO] loading images...")
with open(train_dataset, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    train_data_list = list(csv_reader)

#close file
read_obj.close()

#randomize order of images
shuffle(train_data_list)

#same for test images, types
with open(test_dataset, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    test_data_list = list(csv_reader)


read_obj.close()

(train_data,train_labels)=create_data_arrays_list(train_data_list,image_dimension, verbose=250)

(test_data,test_labels)=create_data_arrays_list(test_data_list,image_dimension,verbose=250)

#Set classNames from train_labels list, rename to eliminate numbers from the front
classNames = [str(x) for x in np.unique(train_labels)]
classNames=['Kanaa',  'Black_Mesa', 'Sosi', 'Dogoszhi', 'Flagstaff', 'Tusayan', 'Kayenta']

train_data = train_data.astype("float")
#train_data = resnet_v2.preprocess_input(train_data)

print("Training data loaded")

#Load test data separately
test_data = test_data.astype("float") 
#test_data = resnet_v2.preprocess_input(test_data)
print("Test data loaded")


# set X,Y values to values for train, test
(trainX, testX, trainY, testY) = (train_data,test_data,train_labels,test_labels)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


    

# compile  model after setting base model layers to be non-trainable
print("Compiling model...")

#Gradient descent optimization algorithm
#opt = RMSprop(learning_rate=0.0025)
opt = RMSprop(learning_rate=0.01)

#Compile, specify loss, optimizer, metrics
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#Add Learning rate scheduler to model callbacks
callbacks=[LearningRateScheduler(step_decay)]

# train the head of the network x epochs; other layers frozen to prevent randomization
print("[INFO] training head...")

epoch_head = 10
model.fit(aug.flow(trainX, trainY, batch_size=image_batch_size),
	validation_data=(testX, testY), epochs=epoch_head,
	steps_per_epoch=len(trainX) // image_batch_size, callbacks=callbacks,verbose=1)


full_model_path = os.path.join(models_dir,'',model_id +'.keras')
#Set callbacks for final run, including saving models based on test accuracy (called val_acc here),
callbacks=[LearningRateScheduler(step_decay),ModelCheckpoint(full_model_path, monitor='val_accuracy', verbose=1,save_best_only=True,
          save_weights_only=False,mode='max',save_freq='epoch')]

# unfreeze the initial set of deep layers  and make them trainable; remember 0-based count!
for layer in baseModel.layers[0:]:
	layer.trainable = True

#for layer in baseModel.layers[0:]:
    #if isinstance(layer, BatchNormalization):
      #layer.trainable = False

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a *very* small learning rate
print("Re-compiling model...")
opt = SGD(learning_rate=0.005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print("Final model compiled")

# train the model again, this time fine-tuning the full set of CONV layers
print("Fine-tuning model...")
epoch_full=40
image_batch_size=8
H=model.fit(aug.flow(trainX, trainY, batch_size=image_batch_size),
	validation_data=(testX, testY), epochs=epoch_full,
	steps_per_epoch=len(trainX) // image_batch_size, callbacks=callbacks, verbose=1)

model_best = load_model(full_model_path)


# evaluate train data
print("[INFO] evaluating train data ...")
predictions = model_best.predict(trainX, batch_size=16)
class_report=classification_report(trainY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames)
class_file=open(full_model_path  + "_train_class_report.txt","w")
class_file.write(class_report)
class_file.close()
print(class_report)
#print(classification_report(np.argmax(testY,axis=1), predictions.argmax(axis=1),target_names=classNames))
print("Confusion matrix")
print(classNames)
con_mat=confusion_matrix(np.argmax(trainY,axis=1), predictions.argmax(axis=1))
#print(confusion_matrix(np.argmax(testY,axis=1), predictions.argmax(axis=1)))
print(con_mat)
#confusion_matrix(np.argmax(testY,axis=1), predictions.argmax(axis=1))
np.savetxt(full_model_path  + "_train_confusion_matrix.csv", con_mat, delimiter=",")


# evaluate test data
print("[INFO] evaluating test data ...")
predictions = model_best.predict(testX, batch_size=16)
class_report=classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames)
class_file=open(full_model_path  + "_test_class_report.txt","w")
class_file.write(class_report)
class_file.close()
print(class_report)
#print(classification_report(np.argmax(testY,axis=1), predictions.argmax(axis=1),target_names=classNames))
print("Confusion matrix")
print(classNames)
con_mat=confusion_matrix(np.argmax(testY,axis=1), predictions.argmax(axis=1))
#print(confusion_matrix(np.argmax(testY,axis=1), predictions.argmax(axis=1)))
print(con_mat)
#confusion_matrix(np.argmax(testY,axis=1), predictions.argmax(axis=1))
np.savetxt(full_model_path  + "_test_confusion_matrix.csv", con_mat, delimiter=",")

# plot the accuracy
plt.style.use("ggplot")
plt.figure(figsize=(11, 8))
plt.plot(np.arange(0, epoch_full), H.history["accuracy"], label="Train accuracy")
plt.plot(np.arange(0, epoch_full), H.history["val_accuracy"], label="Test accuracy")
plt.title(model_id + " Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(full_model_path + "_accuracy_plot.png")
#plt.show()

# plot the training loss
plt.style.use("ggplot")
plt.figure(figsize=(11, 8))
plt.plot(np.arange(0, epoch_full), H.history["loss"], label="Train loss")
plt.plot(np.arange(0, epoch_full), H.history["val_loss"], label="Test loss")
plt.title(model_id + " Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(full_model_path  + "_loss_plot.png")
#plt.show()

