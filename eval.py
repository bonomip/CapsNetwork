import os
import re
import sys
import argparse
sys.path.append("./src")
from setup import Setup # set up model and dataset

#number of checkpoints physically inside log/model directory
no_ckpt = 100

def get_all_datasets(setup, model_id, dataset_version):
    data = []
    x, y, b = setup.load_data(model_id, train=True, version=dataset_version, create=False)
    data = data + [x, y, b]
    for i in range(len(Setup.GEN)):
        x, y, b = setup.load_data(Setup.GEN[i], train=False, version=dataset_version, create=False)
        data = data + [x, y, b]
    return data
#function to make a pretty format
def make_header(name):
    #padding for header
    length = 44
    p1 = int((length-len(name)-2)/2)
    p2 = length-p1-len(name)-2
    return ("-"*p1)+" "+name+" "+("-"*p2)

#setup argument parser
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
parser.add_argument("-f",
                    choices=range(1,no_ckpt),
                    type=int,
                    required=True,
                    metavar="First epoch to be considered")                 
parser.add_argument("-t",
                    choices=range(1,no_ckpt+1),
                    type=int,
                    required=True,
                    metavar="Last epoch to be considered")  
parser.add_argument("-d",
                    default=False,
                    action='store_true')   
parser.add_argument("--model-version",
                    default="_3e-5",
                    type=str) 
parser.add_argument("--no-gpu",
                    default=False,
                    action='store_true') 
args=parser.parse_args()

if args.no_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#model to evaluate
model_id = Setup.GEN[args.model]
#check if range is positive
if (args.t - args.f) < 0:
  raise Exception("Sorry, no numbers below zero")  
# model and dataset version
model_version = args.model_version
dataset_version = "_v1"
# path where result would be saved
file_path = "./results/"+model_id+model_version+".txt"

#init file if first time -- THIS WOULD RESET THE FILE
if args.f == 1:
    file = open(file_path, "w")
    file.write("\n"+make_header(model_id)+"\n")
    file.close()

#create setup object
setup = Setup(debug=args.d, no_gpu_check=args.no_gpu)
# retrieve all datasets
data = get_all_datasets(setup, model_id, dataset_version)
#extract train ones
X_train, y_train, dataset = [data[i] for i in (0, 1, 2)]
#remove train data from array
data = data[3:]
#init model
model = setup.init_model(model_id, model_version, X_train, y_train)

# evaluate accuracy for each epoch on each dataset
for i in range(args.f, args.t+1):
    #array where accuracies would be stored
    # idx -> 0: train set; from 1 to 4 -> test sets
    accuracies = []
    #print current epoch on terminal
    print("---- "+str(i)+"-th Epoch ----")
    #description string for progression bar
    s = "- "+model_id+" train"
    #load ckpt for current epoch
    setup.load_ckpt(model, i)
    #evaluate accuray on train set
    accuracies.append(setup.get_accuracy(model, dataset, setup.get_total_images(X_train), s))
    for j in range(len(Setup.GEN)):
    
        #evaluate accuracy on test sets
        s = "- "+Setup.GEN[j]
        X_test, y_test, testing = [data[i] for i in ((j*3)+0, (j*3)+1, (j*3)+2)]
        accuracies.append(setup.get_accuracy(model, testing, setup.get_total_images(X_test), s))
    
    #convert accuracy to string
    string_list = ["%.4f" % number for number in accuracies]
    #get padding
    pad1 = max(map(len, Setup.GEN))+2
    pad2 = max(map(len, string_list))+2
    #write to file
    file = open(file_path, "a+")
    file.write(make_header(str(i)+"-th EPOCH"))
    file.write("\n~ "+f'{"Train":<{pad1}}:{string_list[0]:>{pad2}}')
    for j in range(len(Setup.GEN)):

        name = Setup.GEN[j].replace('_', ' ')
        file.write("\n~ "+f'{name:<{pad1}}:{string_list[j+1]:>{pad2}}')

    file.write("\n")
    file.close()