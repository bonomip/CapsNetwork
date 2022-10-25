from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np

def pretty_experiment_overview(setup):
    
    no_train_imgs = setup.X_train.shape[0]
    no_test_imgs = setup.X_test.shape[0]
    test_dataset_type = setup.test_dataset_setting
    model_type = setup.train_dataset_setting
    epochs= setup.epochs

    s = "The model was trained with "+str(no_train_imgs)+" "+model_type+" images for "+str(epochs)+" epochs.\n"
    s += "It's going to be tested with "+str(no_test_imgs)+" "+test_dataset_type+" images."
    
    print(s)

def print_image_and_prediction(x, y, p, size):
    print_image(x, "value: "+str(y)+" - prediction: "+str(p), size)

def print_image(x, y, size):
    plt.figure()
    plt.xlabel(y)
    plt.imshow(tf.reshape(x, [1, size, size, 1])[0])

def get_sum_row_matrix_label(matrix):
    res = []
    for c in range(0, 10):
        res.append( str( int( np.sum( matrix[:,c] ) ) ) )
    return res

def get_sum_colum_matrix_label(matrix):
    res = []
    for r in range(0, 10):
        res.append( str( int( np.sum( matrix[r,:] ) ) ) )
    return res

def print_matrix(matrix, matrix_type, model_type, data_type, no_images, col_labels=0, row_labels=0, size_x=10, size_y=10, color="Blues"):
        headers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
        title = "CapsNet= "+model_type+"\nConfusion Matrix= "+matrix_type+", based on "+str(no_images/1000)+"K "+data_type+" images"
        x_label = "Actual Digit"
        y_label = "Predicted Digit"
        
        if row_labels != 0:
            fig, ax = plt.subplots(figsize=(size_x,size_y))
            mat = matrix.copy()
            mat = mat.transpose()
            
            ax = sns.heatmap(mat, annot=True, fmt='g', cmap=color, ax=ax)

            ax.set_title(title+'\n\n', size=size_x*1.9)
            ax.set_xlabel('\n'+y_label, size=size_x*1.5)
            ax.set_ylabel(x_label+' \n', size=size_x*1.5)
            
            ax3 = ax.twiny()
            # ax3.set_aspect("equal")
            ax3.set_xlabel("No. of occurencies\n", size=size_x*1.5)
            ax3.set_xlim([0,ax.get_xlim()[1]])
            ax3.set_xticks(ax.get_xticks())
            ax3.set_xticklabels(row_labels, fontsize=size_x*1)
            ax3.tick_params(top=False)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.spines['left'].set_visible(False)
        else:
            fig, ax = plt.subplots(figsize=(size_x,size_y))
            ax = sns.heatmap(matrix, annot=True, fmt='g', cmap=color, ax=ax)

            ax.set_title(title+'\n\n', size=size_x*1.9)
            ax.set_xlabel('\n'+x_label, size=size_x*1.5)
            ax.set_ylabel(y_label+' \n', size=size_x*1.5)


        if col_labels != 0:            
            ax3 = ax.twiny()
            # ax3.set_aspect("equal")
            ax3.set_xlabel("No. of occurencies\n", size=size_x*1.5)
            ax3.set_xlim([0,ax.get_xlim()[1]])
            ax3.set_xticks(ax.get_xticks())
            ax3.set_xticklabels(col_labels, fontsize=size_x*1)
            ax3.tick_params(top=False)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.spines['left'].set_visible(False)  

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(headers)
        ax.yaxis.set_ticklabels(headers)

        ## Display the visualization of the Confusion Matrix.
        plt.show()
        
def trim(string):
    if string.endswith(".0"):
        return string[0:-2]
    return string
    
def print_confusion_tables(values, columns, index):
    df = pd.DataFrame( values , index=columns, columns=index, dtype=str).T
    df.columns.name = "Digit"
    df = df.applymap(trim)
    return df

def print_accuracy(acc_train, acc_test, train_s, test_s):
    d = {
    "Accuracy" : [np.around(acc_train, decimals=3), np.around(acc_test, decimals=3)],
    "# Images" : [train_s, test_s],
    "Epochs" : [10, 10]
    }

    return pd.DataFrame(d, index=["Train", "Test"])

def show_grid(x, dimx, dimy, title, size=20.):
    fig = plt.figure(figsize=(size, size))
    fig.suptitle(title, fontsize=16)
    grid = ImageGrid( fig, 111,nrows_ncols=(dimy, dimx),axes_pad=0.1, )
    a = []

    for i in range(dimx*dimy):
        j = x[:, :, i]
        a.append(j)

    for ax, im in zip(grid, a):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(im, cmap="gray")

    fig.subplots_adjust(top=0.97) 
    plt.show()

# TODOs adjust print based on weights shape
def print_fixed_network_params(model):
    f = model.layers[0].get_weights()[0] # f.shape: (9, 9, 1, 256)
    show_grid(f[:,:,0,:], 16, 16, "Conv1 256 filters") # show filters
    
    f = model.layers[1].get_weights()[0] # f.shape: (9, 9, 256, 256)
    show_grid(f[:,:,:,0], 16, 16, "Primary Capsule Filter 9x9x256 #1 of 32*8") # first filter 9x9x256
    show_grid(f[:,:,:,0], 16, 16, "Primary Capsule Filter 9x9x256 #2 of 32*8") # first filter 9x9x256
    show_grid(f[:,:,:,0], 16, 16, "Primary Capsule Filter 9x9x256 #3 of 32*8") # first filter 9x9x256    

# TODOs adjust print based on x shape    
def show_conv1_rfm(x, f, b):
    x = tf.nn.conv2d(x,f,strides=[1, 1],padding='VALID')
    x = tf.add(x, b)
    x = tf.nn.relu(x).numpy()
    show_grid(x[0], 16, 16, "Conv1 256 output 32x32x256")

# TODOs adjust print based on x shape    
def show_prim_caps_rfm(x, f, b):
    x = np.array(tf.reshape( x, [1, 32, 32, 256]))
    x = tf.nn.conv2d(x,f,strides=[2, 2],padding='VALID')
    x = tf.add(x, b)
    show_grid(x[0], 8, 32, "Primary Capsule 12x12x8 output, each row is a channel (32 total)", 40.)


def print_network(model, x, y, image_size):    
    
    x = tf.reshape(x, [1, image_size, image_size, 1])
    print_image_and_prediction(x, y, model.predict(x)[0], image_size)

    f = model.layers[0].get_weights()[0]
    b = model.layers[0].get_weights()[1]
    show_conv1_rfm(x, f, b) 

    x = model.convolution(x)
    f = model.layers[1].get_weights()[0]
    b = model.layers[1].get_weights()[1]
    show_prim_caps_rfm(x, f, b)

    x = model.primary_capsule(x)