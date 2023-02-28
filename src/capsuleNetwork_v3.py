import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime

class CapsuleNetwork(tf.keras.Model):

    epsilon=1e-7
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5
    alpha = 0.0005
    learning_rate = 3e-5
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    save_every_epochs = 1
    patience = 10


    def __init__(self, size, no_of_conv_kernels, no_of_primary_caps_channels, 
                    no_of_secondary_capsules, primary_capsule_vector, secondary_capsule_vector, r, id, version, learning_rate=3e-5):
        
        super(CapsuleNetwork, self).__init__()

        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model_id = id
        self.model_version = version
        
        self.no_of_conv3_kernels = no_of_conv_kernels
        self.no_of_conv2_kernels = no_of_conv_kernels / 2
        self.no_of_conv1_kernels = no_of_conv_kernels / 4

        self.no_of_primary_caps_channels = no_of_primary_caps_channels
        self.no_of_secondary_capsules = no_of_secondary_capsules
        self.primary_capsule_vector = primary_capsule_vector
        self.secondary_capsule_vector = secondary_capsule_vector

        self.r = r

        #dense layers size
        self.d1 = 1024
        self.d2 = 2048
        self.d3 = size*size #40*40

        self.no_primary_capsule = 1152

        with tf.name_scope("Variables") as scope:

            self.conv1 = tf.keras.layers.Conv2D(self.no_of_conv1_kernels, [5,5], strides=[1,1], 
                                                        name='ConvolutionLayer1', activation='relu')

            self.conv2 = tf.keras.layers.Conv2D(self.no_of_conv2_kernels, [9,9], strides=[1,1], 
                                                        name='ConvolutionLayer2', activation='relu')

            self.conv3 = tf.keras.layers.Conv2D(self.no_of_conv3_kernels, [9,9], strides=[1,1], 
                                                        name='ConvolutionLayer3', activation='relu')
            
            self.primary_capsule = tf.keras.layers.Conv2D(self.no_of_primary_caps_channels * self.primary_capsule_vector, 
                                                            [9,9], strides=[2,2], name="PrimaryCapsule")

            self.w = tf.Variable(tf.random_normal_initializer()(shape=[1, self.no_primary_capsule, 
                                                                    self.no_of_secondary_capsules, 
                                                                    self.secondary_capsule_vector, self.primary_capsule_vector]), 
                                                                dtype=tf.float32, name="PoseEstimation", trainable=True)
            
            # fully connected layers for image reconstruction
            self.dense_1 = tf.keras.layers.Dense(units = self.d1, activation='relu')
            self.dense_2 = tf.keras.layers.Dense(units = self.d2, activation='relu')
            self.dense_3 = tf.keras.layers.Dense(units = self.d3, activation='sigmoid', dtype='float32')
        
    def build(self, input_shape):
        pass
    
    def _epochs_to_cpkt(self, epochs):
        return int(epochs/self.save_every_epochs)

    def get_checkpoint_path(self):
        return './logs/v3/'+self.model_id+'/model'+self.model_version

    def load_latest(self):
        checkpoint = tf.train.Checkpoint(model=self)
        path = self.get_checkpoint_path()+"/ckpt-"
        ckpt = tf.train.latest_checkpoint(path)
        checkpoint.restore(ckpt)

    def load(self, epochs=-1):
        if epochs==0:
        
            return 0

        #create checkpoint obj binded to model
        checkpoint = tf.train.Checkpoint(model=self)
        #ckpt base path
        path = self.get_checkpoint_path()+"/ckpt-"
        #default is load latest ckpt
        if epochs<0:
    
            checkpoint.restore(tf.train.latest_checkpoint(path))
    
        else:

            #convert epoch to ckpt index
            i = self._epochs_to_cpkt(epochs)
            #load weights into model
            checkpoint.restore(path+str(i))
        
        return 1

    def save_resume(self, path, epoch, wait=0, best=0, best_epoch=0):
        #save information for resume train later
        with open(path, "w") as f:
            #current epoch, wait value, validation accuracy
            f.write(str(epoch)+" "+str(wait)+" "+str(best)+" "+str(best_epoch))
            f.write(" learning_rate:"+str(self.learning_rate))

    def squash(self, s):
        with tf.name_scope("SquashFunction") as scope:
            s_norm = tf.norm(s, axis=-1, keepdims=True)
            return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + self.epsilon)
    
    def safe_norm(self, v, axis=-1, epsilon=1e-7):
        v_ = tf.reduce_sum(tf.square(v), axis = axis, keepdims=True)
        return tf.sqrt(v_ + epsilon)

    def predict(self, x):
        pred = self.safe_norm(self.predict_capsule_output(x))
        pred = tf.squeeze(pred, [1])
        return np.argmax(pred, axis=1)[:,0]

    def loss_function(self, v, reconstructed_image, y, y_image):
        ### margin loss is a reduce_mean .... in the paper this is not specified.
        ### in the paper is only the summ of all the digit loss.
        prediction = self.safe_norm(v)
        prediction = tf.reshape(prediction, [-1, self.no_of_secondary_capsules])
        left_margin = tf.square(tf.maximum(0.0, self.m_plus - prediction))
        right_margin = tf.square(tf.maximum(0.0, prediction - self.m_minus))
        l = tf.add(y * left_margin, self.lambda_ * (1.0 - y) * right_margin)
        margin_loss = tf.reduce_mean(tf.reduce_sum(l, axis=-1))
        y_image_flat = tf.reshape(y_image, [-1, 1600])
        reconstruction_loss = tf.reduce_mean(tf.square(y_image_flat - reconstructed_image))
        loss = tf.add(margin_loss, self.alpha * reconstruction_loss)
        
        return loss

    @tf.function
    def train(self, x,y):
        y_one_hot = tf.one_hot(y, depth=10)
        with tf.GradientTape() as tape:
            v, reconstructed_image = self([x, y_one_hot])
            loss = self.loss_function(v, reconstructed_image, y_one_hot, x)
            
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss

    def train_for_epochs(self, batch, epochs, resume=False, v_batch=0):
        
        #path to checkpoint
        checkpoint_path = self.get_checkpoint_path()
        #path to file used to store values necessary to restore training
        restore_file = checkpoint_path+"/restore.txt"
        #epoch from which start training
        start_epoch = 0
        #early stopping parameters
        wait = 0
        best = 0
        best_epoch = 0

        #if we are traning in rounds
        if resume:

            #if no resume file exist
            if not os.path.exists(restore_file):

                print("No resume file found! Impossible to resume training!")

            else:
                
                #load data from file
                with open(restore_file, "r") as f:

                    a = f.read().split(" ")
                    start_epoch = int(a[0])
                    wait = int(a[1])
                    best = float(a[2])
                    best_epoch = int(a[3])

                #load last weights
                self.load(start_epoch)

        #set up checkpoint object
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.save_counter.assign_add(self._epochs_to_cpkt(start_epoch))

        #epochs loop
        for i in range(start_epoch+1, epochs+1, 1):
        
            with tqdm(total=len(batch)) as pbar:

                pbar.set_description_str("Epoch " + str(i) + "/" + str(epochs))
                pbar.set_postfix({'patience': wait})  
                #train loop
                for X_batch, y_batch in batch:

                    self.train(X_batch,y_batch)
                    pbar.update(1)

                #save checkpoint    
                if i % self.save_every_epochs == 0:
                    
                    pbar.set_postfix_str('saving ckpt...')  
                    checkpoint.save(checkpoint_path+"/ckpt")

                pbar.set_postfix_str('')  

                #early stopping    
                if(v_batch != 0):
                    
                    pbar.set_postfix_str('eval...')  
                    training_sum = 0
                    total_img = 0
                    #evaluate accuracy on validation set
                    for X_batch, y_batch in v_batch:
                        
                        total_img += X_batch.shape[0]
                        training_sum += sum(self.predict(X_batch)==y_batch.numpy())

                    accuracy = training_sum/total_img
                    pbar.set_postfix({'acc': accuracy})
                    wait += 1
                    #if the model is improving
                    if accuracy > best:
                        best_epoch = i
                        best = accuracy
                        wait = 0
                    #if the model is overfitting
                    if wait >= self.patience:

                        pbar.set_postfix_str('early stopped!')
                        break
                    
                self.save_resume(restore_file, i, wait, best, best_epoch)
                
               
            

    @tf.function
    def call(self, inputs):
        input_x, y = inputs
        
        x = self.conv1(input_x) #36x36x64
        x = self.conv2(x) # 28x28x128
        x = self.conv3(x) # 20x20x256
        x = self.primary_capsule(x) # 6x6x256

        with tf.name_scope("CapsuleFormation") as scope:
                                            
            u = tf.reshape(x, (-1,  self.no_of_primary_caps_channels * x.shape[1] * x.shape[2], self.primary_capsule_vector))
            u = tf.expand_dims(u, axis=-2)  
            u = tf.expand_dims(u, axis=-1)     
            u_hat = tf.matmul(self.w, u)   
            u_hat = tf.squeeze(u_hat, [4])
        
        with tf.name_scope("DynamicRouting") as scope:
                                                                       
            b = tf.zeros((input_x.shape[0], self.no_primary_capsule, self.no_of_secondary_capsules, 1)) 
            for i in range(self.r):
                c = tf.nn.softmax(b, axis=-2) 
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
                v = self.squash(s)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4])
                b += agreement
                
        with tf.name_scope("Masking") as scope:
            y = tf.expand_dims(y, axis=-1) # y.shape: (None, 10, 1)
            y = tf.expand_dims(y, axis=1) # y.shape: (None, 1, 10, 1)
            mask = tf.cast(y, dtype=tf.float32) # mask.shape: (None, 1, 10, 1)
            v_masked = tf.multiply(mask, v) # v_masked.shape: (None, 1, 10, 16)
            
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(v_masked, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 160)
            reconstructed_image = self.dense_1(v_)
            reconstructed_image = self.dense_2(reconstructed_image)
            reconstructed_image = self.dense_3(reconstructed_image)
        
        return v, reconstructed_image
    
    @tf.function
    def predict_capsule_output(self, inputs):
        x = self.conv1(inputs) #36x36x64
        x = self.conv2(x) # 28x28x128
        x = self.conv3(x) # 20x20x256
        x = self.primary_capsule(x)
        
        with tf.name_scope("CapsuleFormation") as scope:
            u = tf.reshape(x, (-1, self.no_of_primary_caps_channels * x.shape[1] * x.shape[2], 8))
            u = tf.expand_dims(u, axis=-2)
            u = tf.expand_dims(u, axis=-1)
            u_hat = tf.matmul(self.w, u)
            u_hat = tf.squeeze(u_hat, [4])

        
        with tf.name_scope("DynamicRouting") as scope:
            b = tf.zeros((inputs.shape[0], self.no_primary_capsule, self.no_of_secondary_capsules, 1))
            for i in range(self.r):
                c = tf.nn.softmax(b, axis=-2)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
                v = self.squash(s)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4])
                b += agreement
        return v

    @tf.function
    def regenerate_image(self, inputs):
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(inputs, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector])
            reconstructed_image = self.dense_1(v_)
            reconstructed_image = self.dense_2(reconstructed_image)
            reconstructed_image = self.dense_3(reconstructed_image)
        return reconstructed_image       