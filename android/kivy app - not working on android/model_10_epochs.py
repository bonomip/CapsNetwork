from capsuleNetwork import CapsuleNetwork
import tensorflow as tf
from datetime import datetime

def show_plain_output(single_image):
    #single image is a numpy array 28x28x1
    global model
    return CapsuleNetwork.show_plain_output(model, single_image)

#device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:0':
#    raise SystemError('GPU device not found')
#print('Found GPU at: {}'.format(device_name))

BATCH_SIZE = 64

###### SET UP TRAIN AND TEST DATA

print("\n\n \t\t---- SET UP LOGS ---\n\n")

base_path = "./"

checkpoint_path = base_path+'logs/model/capsule'

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

logdir = base_path+'logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

scalar_logdir = base_path+'logs/scalars/%s' % stamp
file_writer = tf.summary.create_file_writer(scalar_logdir + "/metrics")

print("\n\n \t\t---- SET UP DATA SET ---\n\n")

(X_train, y_train), (X_test , y_test)= tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_train = tf.cast(X_train, dtype=tf.float32)
X_train = tf.expand_dims(X_train, axis=-1)

X_test = X_test / 255.0
X_test = tf.cast(X_test, dtype=tf.float32)
X_test = tf.expand_dims(X_test, axis=-1)

testing_dataset_size = X_test.shape[0]
training_dataset_size = X_train.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size=BATCH_SIZE)

testing = tf.data.Dataset.from_tensor_slices((X_test, y_test))
testing = testing.batch(batch_size=BATCH_SIZE)

tf.summary.trace_on(graph=True, profiler=True)

###### LOAD MODEL

print("\n\n \t\t---- LOAD MODEL ---\n\n")

params = {
    "no_of_conv_kernels": 256,
    "no_of_primary_capsules": 32,
    "primary_capsule_vector": 8,
    "secondary_capsule_vector": 16,
    "r":3,
}

model = CapsuleNetwork(**params)

model.load()

##### BUILD MODEL

print("\n\n \t\t---- BUILD MODEL ---\n\n")

_ = model.train(X_train[:int(BATCH_SIZE/2)],y_train[:int(BATCH_SIZE/2)])
with writer.as_default():
    tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=logdir) 
    
tf.summary.trace_off()
