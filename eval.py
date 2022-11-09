import sys
sys.path.append("./src")
from setup import Setup # set up model and dataset
import argparse
import re

#number of checkpoints physically inside log/model directory
no_ckpt = 16

#function to make a pretty format
def make_header(name):
    #padding for header
    length = 44
    p1 = int((length-len(name)-2)/2)
    p2 = length-p1-len(name)-2
    return ("-"*p1)+" "+name+" "+("-"*p2)

#parse txt file to array -- RETURNS and array where each entry is a dictionary
# --------------------------------- referenced by the name of the dataset
def parse(path):
    with open(path) as f:
        value_list = re.findall('\d*?\.\d+', f.read())
    name_list = Setup.GEN
    name_list.insert("Train")
    step = len(name_list)
    epochs = [value_list[i*step:(i+1)*step] for i in range(int(len(list)/step))]
    array = []
    for i in range(len(epochs)):
    
        entry = {}
        for j in range(step):

            entry[name_list[j]] = epochs[i][j]

        array.append(entry)

    return array

#setup argument parser
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
parser.add_argument("--no-gpu",
                    default=False,
                    action='store_true') 
args=parser.parse_args()

#model to evaluate
model_id = Setup.GEN[args.model]
#check if range is positive
if (args.t - args.f) < 0:
  raise Exception("Sorry, no numbers below zero")  
#model and dataset version
model_version = "_v1"
dataset_version = "_v1"
#path where result would be saved
file_path = "./"+model_id+".txt"

#init file if first time -- THIS WOULD RESET THE FILE
if args.f == 1:
    file = open(file_path, "w")
    file.write("\n"+make_header(model_id)+"\n")
    file.close()

# init desired model, and retrive it's train dataset
setup = Setup(debug=args.d, no_gpu_check=args.no_gpu)
X_train, y_train, dataset = setup.load_data(model_id, train=True, version=dataset_version, create=False)
model = setup.init_model(model_id, model_version)

# evaluate accuracy for each epoch on each dataset
for i in range(args.f, args.t+1):
    #array where accuracies would be stored
    # idx 0: train set
    # idx from 1 to 4: test sets
    accuracies = []
    #string for progression bar
    s2 = " @ "+str(i)+"-th Epoch"
    s = "on "+model_id+s2
    #load ckpt for current epoch
    setup.load_ckpt(model, X_train, y_train, i)
    #evaluate accuray on train set
    accuracies.append(setup.get_accuracy(model, dataset, setup.get_total_images(X_train), s))
    for j in range(len(Setup.GEN)):
    
        #evaluate accuracy on test sets
        s = "on "+Setup.GEN[j]+s2
        X_test, y_test, testing = setup.load_data(Setup.GEN[j], train=False, version=dataset_version, create=False)
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