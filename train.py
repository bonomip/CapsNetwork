import sys
import argparse
sys.path.append("./src")
from setup import Setup # set up model and dataset

parser=argparse.ArgumentParser()
parser.add_argument("--model",
                    metavar="ID",
                    choices=range(0,len(Setup.GEN)), 
                    type=int,
                    required=True, 
                    help="0="+str(Setup.GEN[0])+"; "+
                        "1="+str(Setup.GEN[1])+"; "+
                        "2="+str(Setup.GEN[2])+"; "+
                        "3="+str(Setup.GEN[3])+"; "+
                        "4="+str(Setup.GEN[4])+"; "+
                        "5="+str(Setup.GEN[5]))
parser.add_argument("--resume",
                    default=False,
                    action='store_true',
                    help="Resume training")               
parser.add_argument("-d",
                    default=False,
                    help="Enable debug",
                    action='store_true')
parser.add_argument("--no-gpu",
                    default=False,
                    action='store_true') 
parser.add_argument("--learning-rate",
                    help="Set learning rate",
                    default=3e-5,
                    type=float) 
args=parser.parse_args()

#learning rate
learning_rate = args.learning_rate

#model to evaluate
model_id = Setup.GEN[args.model]
#model and dataset version
model_version = "_"+str(learning_rate) # the type of training the model has
dataset_version = "_v1" # the dataset version
#validation set; used only for ealy stopping
validation=0

#load model ckpt and dataset
setup = Setup(debug=args.d, no_gpu_check=args.no_gpu)
X_train, y_train, dataset = setup.load_data(model_id, train=True, version=dataset_version, create=False)
model = setup.init_model(model_id, model_version, X_train, y_train, learning_rate)

#load validation for early stopping (we are using test set)
x, y, validation = setup.load_data(model_id, train=False, version=dataset_version, create=False)

#launch training
model = setup.train_model(model, dataset, epochs=100, resume=args.resume, v_batch=validation)
