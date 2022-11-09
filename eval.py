import sys
sys.path.append("./src")
from setup import Setup # set up model and dataset
import argparse

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
args=parser.parse_args()

##########
########## MODEL TO EVALUATE
##########
#
#
model_id = Setup.GEN[args.model]
#
#
##########
########## NO. OF CHECKPONT
##########
#
#
tot_ckpt = int(int(args.epochs))
#
#
##########
########## MODEL & DATA VERSION
##########
#
#
model_version = "_v1"
dataset_version = "_v1"
#
#
##########
##########
##########
#
#
#

setup = Setup(debug=True)
X_train, y_train, dataset = setup.load_data(model_id, train=True, version=dataset_version, create=False)
model = setup.init_model(model_id, model_version)

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

print("\n--------- MODEL "+model_id+model_version+" ---------\n")
for i in range(len(s_epochs)):
    print("--------- "+str(i+1)+"-th EPOCH ---------")
    print("~ train = "+str(s_epochs[i][0]*100))
    for j in range(len(Setup.GEN)):
        print("~ "+Setup.GEN[j]+" = "+str(s_epochs[i][j+1]*100))
print("-----------------------------------------")