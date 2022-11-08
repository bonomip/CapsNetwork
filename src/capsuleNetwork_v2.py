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
    optimizer = tf.keras.optimizers.Adam()
    save_every_epochs = 1


    def __init__(self, size, no_of_conv_kernels, no_of_primary_caps_channels, 
                    no_of_secondary_capsules, primary_capsule_vector, secondary_capsule_vector, r, id, version):
        
        super(CapsuleNetwork, self).__init__()

        self.model_id = id
        self.model_version = version
        
        self.no_of_conv_kernels = no_of_conv_kernels
        self.no_of_primary_caps_channels = no_of_primary_caps_channels
        self.no_of_secondary_capsules = no_of_secondary_capsules
        self.primary_capsule_vector = primary_capsule_vector
        self.secondary_capsule_vector = secondary_capsule_vector

        self.r = r

        #dense layers size
        self.d1 = 512
        self.d2 = 1024
        self.d3 = size*size

        self.no_primary_capsule = 0

        if size == 28:
            self.no_primary_capsule = 1152
        if size == 40:
            self.no_primary_capsule = 4608 

        
        with tf.name_scope("Variables") as scope:

            self.convolution = tf.keras.layers.Conv2D(self.no_of_conv_kernels, [9,9], strides=[1,1], 
                                                        name='ConvolutionLayer', activation='relu')
            
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
        return './logs/'+self.model_id+'/model'+self.model_version

    def load_latest(self):
        checkpoint = tf.train.Checkpoint(model=self)
        path = self.get_checkpoint_path()+"/ckpt-"
        ckpt = tf.train.latest_checkpoint(path)
        checkpoint.restore(ckpt)

    def load(self, epochs=0):
        if epochs==0:
            return self.load_latest()
    
        i = self._epochs_to_cpkt(epochs)

        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(self.get_checkpoint_path()+"/ckpt-"+str(i))

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

    def train(self, x,y):
        y_one_hot = tf.one_hot(y, depth=10)
        with tf.GradientTape() as tape:
            v, reconstructed_image = self([x, y_one_hot])
            loss = self.loss_function(v, reconstructed_image, y_one_hot, x)
            
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss

    def gerAccuracy(self, test_database):
        test_sum = 0
        for X_batch, y_batch in test_database:
            
            test_sum += sum(self.predict(X_batch)==y_batch.numpy())

        print(test_sum/test_database[0])

    def train_for_epochs(self, batch, no_img, epochs, start_epochs=0):
        
        checkpoint_path = self.get_checkpoint_path()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = './logs/'+self.model_id+'/scalars'+self.model_version+'/%s' % stamp
        file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        checkpoint = tf.train.Checkpoint(model=self)

        checkpoint.save_counter.assign_add(self._epochs_to_cpkt(epochs))

        losses = []
        accuracy = []
        for i in range(start_epochs+1, epochs+1, 1):

            loss = 0
            with tqdm(total=len(batch)) as pbar:

                description = "Epoch " + str(i) + "/" + str(epochs)
                pbar.set_description_str(description)
                for X_batch, y_batch in batch:

                    loss += self.train(X_batch,y_batch)
                    pbar.update(1)

                loss /= len(batch)
                losses.append(loss.numpy())
                training_sum = 0
                print_statement = "Loss :" + str(loss.numpy()) + " Evaluating Accuracy ..."
                pbar.set_postfix_str(print_statement)
                for X_batch, y_batch in batch:
                    
                    training_sum += sum(self.predict(X_batch)==y_batch.numpy())
                
                accuracy.append(training_sum/no_img)

                with file_writer.as_default():
                    tf.summary.scalar('Loss', data=loss.numpy(), step=i)
                    tf.summary.scalar('Accuracy', data=accuracy[-1], step=i)
                
                print_statement = "Loss :" + str(loss.numpy()) + " Accuracy :" + str(accuracy[-1])

                if i % self.save_every_epochs == 0:
                    print_statement += ' Checkpoint Saved'
                    checkpoint.save(checkpoint_path+"/ckpt")
                
                pbar.set_postfix_str(print_statement)  

    @tf.function
    def call(self, inputs):
        input_x, y = inputs
        
        x = self.convolution(input_x) # 32x32x256
        x = self.primary_capsule(x) # 12x12x256(=8x32)

        with tf.name_scope("CapsuleFormation") as scope:
                                            # u.shape: (None, 4608, 8)
            u = tf.reshape(x, (-1,  self.no_of_primary_caps_channels * x.shape[1] * x.shape[2], self.primary_capsule_vector))
            u = tf.expand_dims(u, axis=-2)  # u.shape: (None, 4608, 1, 8)
            u = tf.expand_dims(u, axis=-1)  # u.shape: (None, 4608, 1, 8, 1)
                                            # w.shape: ( 1, 4608, 10, 16, 8 )
                                            # u.shape: ( None, 4608,  1,  8, 1 )
            u_hat = tf.matmul(self.w, u)    # u_hat.shape: ( None, 4608, 10, 16, 1 )
            u_hat = tf.squeeze(u_hat, [4])  # u_hat.shape: (None, 4608, 10, 16 )
        
        with tf.name_scope("DynamicRouting") as scope:
                                                       # 10                  
            b = tf.zeros((input_x.shape[0], self.no_primary_capsule, self.no_of_secondary_capsules, 1)) # b.shape: (None, 4608, 10, 1)
            for i in range(self.r): # self.r = 3
                c = tf.nn.softmax(b, axis=-2) # c.shape: (None, 4608, 10, 1)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) # s.shape: (None, 1, 10, 16)
                v = self.squash(s) # v.shape: (None, 1, 10, 16)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
                # Before matmul following intermediate shapes are present, they are not assigned to a variable but just for understanding the code.
                # u_hat.shape (Intermediate shape) : (None, 4608, 10, 16, 1)
                # v.shape (Intermediate shape): (None, 1, 10, 16, 1)
                # Since the first parameter of matmul is to be transposed its shape becomes:(None, 4608, 10, 1, 16)
                # Now matmul is performed in the last two dimensions, and others are broadcasted
                # Before squeezing we have an intermediate shape of (None, 4608, 10, 1, 1)
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
        x = self.convolution(inputs)
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