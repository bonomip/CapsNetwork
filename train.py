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
parser.add_argument("--resume",
                    default=False,
                    action='store_true',
                    help="Resume training")               
parser.add_argument("--model-version",
                    default="_v1",
                    type=str) 
parser.add_argument("--dataset-version",
                    default="_v1",
                    type=str)
parser.add_argument("-d",
                    default=False,
                    help="Enable debug",
                    action='store_true')   
parser.add_argument("--no-gpu",
                    default=False,
                    action='store_true') 
parser.add_argument("--early-stopping",
                    default=False,
                    help="Enable early stopping",
                    action='store_true') 
args=parser.parse_args()

#model to evaluate
model_id = Setup.GEN[args.model]
#model and dataset version
model_version = args.model_version
dataset_version = args.dataset_version
#validation set; used only for ealy stopping
validation=0

#load model ckpt and dataset
setup = Setup(debug=args.d, no_gpu_check=args.no_gpu)
X_train, y_train, dataset = setup.load_data(model_id, train=True, version=dataset_version, create=False)
model = setup.init_model(model_id, model_version, X_train, y_train)
#load validation set for early stopping
if args.early_stopping:

    x, y, validation = setup.load_data(model_id, train=False, version=dataset_version, create=False)

#launch training
model = setup.train_model(model, dataset, epochs=50, resume=args.resume, v_batch=validation)