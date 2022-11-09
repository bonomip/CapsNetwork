import sys
sys.path.append("./src")
from setup import Setup # set up model and dataset
import argparse

#number of checkpoints physically inside log/model directory
no_ckpt = 16

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
parser.add_argument("--epochs",
                    choices=range(1,no_ckpt+1),
                    type=int,
                    required=True,
                    metavar="No. EPOCHS")  
parser.add_argument("-d",
                    default=False,
                    action='store_true')   
args=parser.parse_args()

#model to evaluate
model_id = Setup.GEN[args.model]
#how many ckpt -- usually max ~ 50
tot_ckpt = int(int(args.epochs))
#model and dataset version
model_version = "_v1"
dataset_version = "_v1"

# init desired model, and retrive it's train dataset
setup = Setup(debug=args.d)
X_train, y_train, dataset = setup.load_data(model_id, train=True, version=dataset_version, create=False)
model = setup.init_model(model_id, model_version)

# evaluate accuracy for each epoch on each dataset
s_epochs = []
for i in range(1, tot_ckpt+1):
    s2 = " @ "+str(i)+"-th Epoch"
    s = "on "+model_id+s2
    setup.load_ckpt(model, X_train, y_train, i)
    accuracies = [] # @ 0 = train, @ 1-4 = test
    accuracies.append(setup.get_accuracy(model, dataset, setup.get_total_images(X_train), s))
    for j in range(len(Setup.GEN)):
    
        s = "on "+Setup.GEN[j]+s2
        X_test, y_test, testing = setup.load_data(Setup.GEN[j], train=False, version=dataset_version, create=False)
        accuracies.append(setup.get_accuracy(model, testing, setup.get_total_images(X_test), s))
    
    s_epochs.append(accuracies)

# display the result
print("\n--------- MODEL "+model_id+model_version+" ---------\n")
for i in range(len(s_epochs)):
    print("--------- "+str(i+1)+"-th EPOCH ---------")
    print("~ train = "+str(s_epochs[i][0]*100))
    for j in range(len(Setup.GEN)):
        print("~ "+Setup.GEN[j]+" = "+str(s_epochs[i][j+1]*100))
print("-----------------------------------------")