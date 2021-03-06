from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.metrics import categorical_accuracy as accuracy
from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_json
from keras.objectives import categorical_crossentropy
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from tensorflow.examples.tutorials.mnist import input_data
# import simplejson as sj
import tensorflow as tf

def tf_model():
    sess = tf.Session()
    K.set_session(sess)
    print(K.learning_phase())

    # This placeholder will contain our input digits, as flat vectors
    img = tf.placeholder(tf.float32, shape=(None,784))

    # Keras layers can be called on TensorFlow tensors:
    x = Dense(128, activation='relu')(img) # fully-connected layer with 128 units and ReLU activation
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(10, activation='softmax')(x) # output layer with 10 units and a softmax activation

    labels = tf.placeholder(tf.float32, shape=(None, 10))

    loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Initialize all variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Run training loop
    with sess.as_default():
        for i in range(100):
            print(K.learning_phase())
            batch = mnist_data.train.next_batch(50)
            train_step.run(feed_dict={img: batch[0],
                                      labels: batch[1],
                                      K.learning_phase(): 1})

    acc_value = accuracy(labels, preds)
    with sess.as_default():
        print(acc_value.eval(feed_dict={img:mnist_data.test.images,
                                        labels: mnist_data.test.labels}))

def classify_img_vgg(input_img_path):
    sess = tf.Session()
    model = VGG16()
    print(model.summary())
    # plot_model(model, to_file='vgg.png')

    # load an imag from file
    image = load_img(input_img_path, target_size=(224, 224))

    # convert the image pixels to a numpy array
    image = img_to_array(image)

    # reshape data for the model
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    # prepare the image for the VGG model
    image = preprocess_input(image)

    # predict the probability across all output classes
    yhat = model.predict(image)

    # convert the probabilities to class labels
    label = decode_predictions(yhat)

    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]

    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))

def classify_img_vgg2(input_img_path):
    sess = tf.Session()

    # model = VGG16()
    # vgg_json = model.to_json()
    # del model

    # with open('vgg_keras_model.json', 'w') as out:
    #     out.write(sj.dumps(sj.loads(vgg_json), indent=4))
    # model = model_from_json(vgg_json)

    input_shape = _obtain_input_shape(None,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=True)
    img_input = Input(shape=input_shape)

    # My attempt on building the VGG16 architecture
    # For some reason this produces the wrong prediction compared to the reference
    #  implementation, most likely because of the lack of an input layer that uses
    #  _obtain_input_shape()
    # model = Sequential()

    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_1'))
    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_2'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_1'))
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_2'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_1'))
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_2'))
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_1'))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_2'))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1'))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2'))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))

    # model.add(Flatten(name='flatten'))
    # model.add(Dense(4096, name='fc_1'))
    # model.add(Dense(4096, name='fc_2'))
    # model.add(Dense(1000, activation='softmax', name='softmax'))

    # Adopted from fchollet's implementation. Uses a tensor that is later on passed to
    #  a keras Model
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='my_vgg16')

    model.load_weights('/home/mark/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    print(model.summary())
    # plot_model(model, to_file='vgg.png')

    # load an imag from file
    image = load_img(input_img_path, target_size=(224, 224))

    # convert the image pixels to a numpy array
    image = img_to_array(image)

    # reshape data for the model
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    # prepare the image for the VGG model
    image = preprocess_input(image)

    # predict the probability across all output classes
    yhat = model.predict(image)

    # convert the probabilities to class labels
    label = decode_predictions(yhat)

    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]

    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))

def main():
    classify_img_vgg2('mug.jpg')

if __name__ == '__main__':
    main()
