import tensorflow as tf
import numpy as np

class CapsuleNetwork(tf.keras.Model):

    epsilon=1e-7
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5
    alpha = 0.0005
    epochs = 10 # 50
    no_of_secondary_capsules = 10
    optimizer = tf.keras.optimizers.Adam()


    def __init__(self, no_of_conv_kernels, no_of_primary_capsules, primary_capsule_vector, secondary_capsule_vector, r):
        super(CapsuleNetwork, self).__init__()
        self.no_of_conv_kernels = no_of_conv_kernels
        self.no_of_primary_capsules = no_of_primary_capsules
        self.primary_capsule_vector = primary_capsule_vector
        self.secondary_capsule_vector = secondary_capsule_vector
        self.r = r
        
        with tf.name_scope("Variables") as scope:
            # creates convolutional layer with 256 9x9 kernels, stride 1, activation relu ( max(n,0) )
            # the model learns 256 9x9 filters and 256 biases
            # applies the filter then adds the bias and applies relu
                                                      # 256
            self.convolution = tf.keras.layers.Conv2D(self.no_of_conv_kernels, [9,9], strides=[1,1], name='ConvolutionLayer', activation='relu')
            
            # learns 32 * 8 (= 252) filters with kernel 9x9 and stride 2
                                                         #   32                       * 8
            self.primary_capsule = tf.keras.layers.Conv2D(self.no_of_primary_capsules * self.primary_capsule_vector, [9,9], strides=[2,2], name="PrimaryCapsule")
            
            # creates trainable tensor: shape=[1, 1152, 10, 16, 8])
            # there are 32 capsule each of those has 36 (6*6) 8D vector (6*6*32 = 1152 => 8D Vectors)
                                                                          #      10                             16                                 8
            self.w = tf.Variable(tf.random_normal_initializer()(shape=[1, 1152, self.no_of_secondary_capsules, self.secondary_capsule_vector, self.primary_capsule_vector]), dtype=tf.float32, name="PoseEstimation", trainable=True)
            
            # fully connected layers for image reconstruction
            self.dense_1 = tf.keras.layers.Dense(units = 512, activation='relu')
            self.dense_2 = tf.keras.layers.Dense(units = 1024, activation='relu')
            self.dense_3 = tf.keras.layers.Dense(units = 784, activation='sigmoid', dtype='float32')
        
    def build(self, input_shape):
        pass
    

    def save(self):
        #  save the current trained module
        self.save_weights('./saved_model/weights/capsule_network_weights')
    
    def load(self):
        self.load_weights('./saved_model/weights/capsule_network_weights'); 

    def squash(self, s):
        with tf.name_scope("SquashFunction") as scope:
            s_norm = tf.norm(s, axis=-1, keepdims=True)
            return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + self.epsilon)
    
    def safe_norm(v, axis=-1, epsilon=1e-7):
        v_ = tf.reduce_sum(tf.square(v), axis = axis, keepdims=True)
        return tf.sqrt(v_ + epsilon)

    def show_plain_output(model, single_image):
        x = tf.reshape(single_image, [1, 28, 28, 1])
        pred = CapsuleNetwork.safe_norm(model.predict_capsule_output(x))
        return pred

    def predict(model, x):
        pred = CapsuleNetwork.safe_norm(model.predict_capsule_output(x))
        pred = tf.squeeze(pred, [1])
        return np.argmax(pred, axis=1)[:,0]

    def loss_function(self, v, reconstructed_image, y, y_image):
        prediction = CapsuleNetwork.safe_norm(v)
        prediction = tf.reshape(prediction, [-1, self.no_of_secondary_capsules])
        
        left_margin = tf.square(tf.maximum(0.0, self.m_plus - prediction))
        right_margin = tf.square(tf.maximum(0.0, prediction - self.m_minus))
        
        l = tf.add(y * left_margin, self.lambda_ * (1.0 - y) * right_margin)
        
        margin_loss = tf.reduce_mean(tf.reduce_sum(l, axis=-1))
        
        y_image_flat = tf.reshape(y_image, [-1, 784])
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

    @tf.function
    def call(self, inputs):
        input_x, y = inputs
        # input_x.shape: (None, 28, 28, 1)
        # y.shape: (None, 10)
        
        x = self.convolution(input_x) #x.shape: (None, 20, 20, 256)
        x = self.primary_capsule(x) #x.shape: (None, 6, 6, 256) => 6*6*8*32 / 8 = 1152
        
        #there are 32 capsule with dimension 8, each of those has 36 (=6*6) 8D vector (36*32 = 1152 => 8D Vectors)
        
        with tf.name_scope("CapsuleFormation") as scope:
            #converting output from shape (None, 6, 6, 256) to shape (None, 1152, 8)
            # from 256 2D images to => 1152 8D vector
                                   # 32                        * 6          * 6            8     
            u = tf.reshape(x, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], self.primary_capsule_vector)) # u.shape: (None, 1152, 8)
            u = tf.expand_dims(u, axis=-2) # u.shape: (None, 1152, 1, 8)
            u = tf.expand_dims(u, axis=-1) # u.shape: (None, 1152, 1, 8, 1)
            
                                                  # w.shape: ( 1, 1152, 10, 16, 8 )
                                               # u.shape: ( None, 1152,  1,  8, 1 )
            u_hat = tf.matmul(self.w, u)   # u_hat.shape: ( None, 1152, 10, 16, 1 )
            u_hat = tf.squeeze(u_hat, [4])  # u_hat.shape: (None, 1152, 10, 16 )
        
        with tf.name_scope("DynamicRouting") as scope:
                                                       # 10                  
            b = tf.zeros((input_x.shape[0], 1152, self.no_of_secondary_capsules, 1)) # b.shape: (None, 1152, 10, 1)
            for i in range(self.r): # self.r = 3
                c = tf.nn.softmax(b, axis=-2) # c.shape: (None, 1152, 10, 1)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) # s.shape: (None, 1, 10, 16)
                v = self.squash(s) # v.shape: (None, 1, 10, 16)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
                # Before matmul following intermediate shapes are present, they are not assigned to a variable but just for understanding the code.
                # u_hat.shape (Intermediate shape) : (None, 1152, 10, 16, 1)
                # v.shape (Intermediate shape): (None, 1, 10, 16, 1)
                # Since the first parameter of matmul is to be transposed its shape becomes:(None, 1152, 10, 1, 16)
                # Now matmul is performed in the last two dimensions, and others are broadcasted
                # Before squeezing we have an intermediate shape of (None, 1152, 10, 1, 1)
                b += agreement
                
        with tf.name_scope("Masking") as scope:
            y = tf.expand_dims(y, axis=-1) # y.shape: (None, 10, 1)
            y = tf.expand_dims(y, axis=1) # y.shape: (None, 1, 10, 1)
            mask = tf.cast(y, dtype=tf.float32) # mask.shape: (None, 1, 10, 1)
            v_masked = tf.multiply(mask, v) # v_masked.shape: (None, 1, 10, 16)
            
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(v_masked, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 160)
            reconstructed_image = self.dense_1(v_) # reconstructed_image.shape: (None, 512)
            reconstructed_image = self.dense_2(reconstructed_image) # reconstructed_image.shape: (None, 1024)
            reconstructed_image = self.dense_3(reconstructed_image) # reconstructed_image.shape: (None, 784)
        
        return v, reconstructed_image
    
    @tf.function
    def predict_capsule_output(self, inputs):
        x = self.convolution(inputs) # x.shape: (None, 20, 20, 256)
        x = self.primary_capsule(x) # x.shape: (None, 6, 6, 256)
        
        with tf.name_scope("CapsuleFormation") as scope:
            u = tf.reshape(x, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 8)) # u.shape: (None, 1152, 8)
            u = tf.expand_dims(u, axis=-2) # u.shape: (None, 1152, 1, 8)
            u = tf.expand_dims(u, axis=-1) # u.shape: (None, 1152, 1, 8, 1)
            u_hat = tf.matmul(self.w, u) # u_hat.shape: (None, 1152, 10, 16, 1)
            u_hat = tf.squeeze(u_hat, [4]) # u_hat.shape: (None, 1152, 10, 16)

        
        with tf.name_scope("DynamicRouting") as scope:
            b = tf.zeros((inputs.shape[0], 1152, self.no_of_secondary_capsules, 1)) # b.shape: (None, 1152, 10, 1)
            for i in range(self.r): # self.r = 3
                c = tf.nn.softmax(b, axis=-2) # c.shape: (None, 1152, 10, 1)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) # s.shape: (None, 1, 10, 16)
                v = self.squash(s) # v.shape: (None, 1, 10, 16)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
                # Before matmul following intermediate shapes are present, they are not assigned to a variable but just for understanding the code.
                # u_hat.shape (Intermediate shape) : (None, 1152, 10, 16, 1)
                # v.shape (Intermediate shape): (None, 1, 10, 16, 1)
                # Since the first parameter of matmul is to be transposed its shape becomes:(None, 1152, 10, 1, 16)
                # Now matmul is performed in the last two dimensions, and others are broadcasted
                # Before squeezing we have an intermediate shape of (None, 1152, 10, 1, 1)
                b += agreement
        return v

    @tf.function
    def regenerate_image(self, inputs):
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(inputs, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 160)
            reconstructed_image = self.dense_1(v_) # reconstructed_image.shape: (None, 512)
            reconstructed_image = self.dense_2(reconstructed_image) # reconstructed_image.shape: (None, 1024)
            reconstructed_image = self.dense_3(reconstructed_image) # reconstructed_image.shape: (None, 784)
        return reconstructed_image