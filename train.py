import sys
import argparse
sys.path.append("./src")
from setup import Setup # set up model and dataset

parser=argparse.ArgumentParser()
parser.add_argument("--model",
                    metavar="ID",
                    choices=range(0,4), 
                    type=int,
                    required=True, 
                    help="0="+str(Setup.GEN[0])+"; "+
                        "1="+str(Setup.GEN[1])+"; "+
                        "2="+str(Setup.GEN[2])+"; "+
                        "3="+str(Setup.GEN[3]))
parser.add_argument("-f",
                    choices=range(0, 49),
                    type=int,
                    required=True,
                    metavar="First epoch to be considered")
parser.add_argument("--patience",
                    choices=range(0, Setup.get_model_patience()),
                    type=int,
                    required=True,
                    metavar="First epoch to be considered")               
parser.add_argument("--model-version",
                    default="_v1",
                    type=str) 
parser.add_argument("--dataset-version",
                    default="_v1",
                    type=str)
parser.add_argument("-d",
                    default=False,
                    action='store_true')   
parser.add_argument("--no-gpu",
                    default=False,
                    action='store_true') 
parser.add_argument("--early-stopping",
                    default=False,
                    action='store_true') 
args=parser.parse_args()

#model to evaluate
model_id = Setup.GEN[args.model]
#model and dataset version
model_version = args.model_version
dataset_version = args.dataset_version
#checkpoint to load and start train from
last_epoch = args.f

#load model ckpt and dataset
setup = Setup(debug=args.d, no_gpu_check=args.no_gpu)
X_train, y_train, dataset = setup.load_data(model_id, train=True, version=dataset_version, create=False)
model = setup.init_model(model_id, model_version)
model = setup.load_ckpt(model, X_train, y_train, epochs=last_epoch)

#load validation set for early stopping
testing = 0
if args.early_stopping:

    X_test, y_test, testing = setup.load_data(model_id, train=False, version=dataset_version, create=False)

model = setup.train_model(model, dataset, epochs=50, start_epoch=last_epoch, v_batch=testing, start_patience=args.patience)