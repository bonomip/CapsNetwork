import numpy as np
import tensorflow as tf

# for project class
import sys
sys.path.append("./src")

from setup import Setup # set up model and dataset


train_dataset_type = Setup.d_k[1]
test_dataset_type = Setup.d_k[0]
sould_be_trained=True
version = "_v1"
should_create_dataset=False
epochs=20

setup = Setup(train_cfg=train_dataset_type, test_cfg=test_dataset_type, should_be_trained=sould_be_trained, 
                  should_create_dataset=should_create_dataset, model_v=version, epochs=epochs, debug=True)

model = setup.get_model()
epochs = model.get_epochs()

X_train, y_train = setup.get_train_images()
X_test, y_test = setup.get_test_images()
no_train_images = setup.get_no_train_images()
no_test_images = setup.get_no_test_images()
dataset = setup.get_dataset()
testing = setup.get_testing()

model.load(train_dataset_type, version, 10)
model.load(train_dataset_type, version, 20)


