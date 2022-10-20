from pyexpat.errors import XML_ERROR_TAG_MISMATCH
import numpy as np
import tensorflow as tf

## project class import

import capsuleNetwork_v2 as capsNet
import affNIST
import caffNIST

class Setup:

    d_k = ["MNIST", "CUSTOM_AFFNIST", "AFFNIST"] #dataset keys
    m_k = ["M40", "CA", "A", "CAWS"] #model keys

    #architecture params
    params = {
        "size": 40,
        "no_of_conv_kernels": 256,
        "no_of_primary_caps_channels": 32,
        "no_of_secondary_capsules": 10,
        "primary_capsule_vector": 8,
        "secondary_capsule_vector": 16,
        "r":3,
    }

    def __init__(self, train_cfg=d_k[1], test_cfg=d_k[1], model_cfg=m_k[1], model_v="_v1", 
                    db_v="_v1", should_be_trained=False, epochs=10, should_create_dataset=False):

        self.train_dataset_setting = train_cfg
        self.test_dataset_setting = test_cfg
        self.model_settings = model_cfg
        self.database_version = db_v
        self.model_version = model_v
        self.train_model = should_be_trained
        self.create_dataset = should_create_dataset
        self.epochs = epochs

        self.check_for_gpu()

        self.train_batch, self.test_batch, self.X_train, self.y_train, self.X_test, self.y_test = self.init_dataset()

        self.model = self.init_model()

    def switch_dataset(self, string, train):
        X_ = 0
        y_ = 0
        
        if string == self.d_k[0]: # MINST
            
            a,b,c,d = 0
            s = "training" if train else "test"
            
            print("Load MINST "+s+" dataset from keras... ")
            (a, b), (c , d)= tf.keras.datasets.mnist.load_data()

            if(train):
                
                X_ = a
                y_ = b
                
            else:
                
                X_ = c
                y_ = d
                
        elif string == self.d_k[1]: # CUSTOM AFFNIST
            
            if(self.create_dataset):
                
                (X_, y_) = caffNIST.create_custom_affnist(train)
                
            else:
                
                (X_, y_) = caffNIST.load(self.database_version, train)

                    
        elif string == self.d_k[2]: # AFFNIST
            
            (X_, y_) = affNIST.load(train)            
        

        return X_, y_

    def init_dataset(self):
        BATCH_SIZE = 64

        (X_train, y_train) = self.switch_dataset(self.train_dataset_setting, train=True)
        (X_test, y_test) = self.switch_dataset(self.test_dataset_setting, train=False)
        
        print("Processing dataset... ")

        X_train = X_train / 255.0
        X_train = tf.cast(X_train, dtype=tf.float32)
        X_train = tf.expand_dims(X_train, axis=-1)

        X_test = X_test / 255.0
        X_test = tf.cast(X_test, dtype=tf.float32)
        X_test = tf.expand_dims(X_test, axis=-1)

        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size=BATCH_SIZE)

        testing = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        testing = testing.batch(batch_size=BATCH_SIZE)

        return dataset, testing, X_train, y_train, X_test, y_test

    def init_model(self):
        
        print("Creating model... ")
        model = capsNet.CapsuleNetwork(**self.params)
        
        if(self.train_model):

            print("Start training model...")
            model.train_for_epochs()

            print("Saving model... ")
            model.save(self.model_settings, self.model_version, self.epochs)
        else:

            print("Loading model... ")
            model.load(self.model_settings, self.model_version, self.epochs)
            _ = model.train(self.X_train[:32],self.y_train[:32])

        return model

    def check_for_gpu(self):
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    ########################## GET OBJECTS

    def get_model(self):
        return self.model    
    
    def get_dataset(self):
        return self.train_batch
    
    def get_testing(self):
        return self.test_batch
    
    def get_train_images(self):
            return self.X_train, self.y_train

    def get_test_images(self):
            return self.X_test, self.y_test