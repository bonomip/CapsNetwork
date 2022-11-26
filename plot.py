# importing the required module
import re
import sys
import numpy as np
import argparse
sys.path.append("./src")
from setup import Setup # set up model and dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#parse txt file to array -- RETURNS and array of epochs, where each entry is a dictionary
# --------------------------------- referenced by the name of the dataset
def parse(path):
    with open(path) as f:
        value_list = re.findall('\d*?\.\d+', f.read())
    name_list = Setup.GEN.copy()
    name_list.insert(0, "Train")
    step = len(name_list)
    epochs = [value_list[i*step:(i+1)*step] for i in range(int(len(value_list)/step))]
    array = []
    for i in range(len(epochs)):
    
        entry = {}
        for j in range(step):

            entry[name_list[j]] = float(epochs[i][j])

        array.append(entry)

    return array

#setup argument parser
parser=argparse.ArgumentParser()              
parser.add_argument("--model-version",
                    default="_v1",
                    type=str)
parser.add_argument("--exclude-affnist",
                    default=False,
                    action='store_true')
args=parser.parse_args()    


#models to plot
            #MNIST       #affNIST     #my affNIST   #my affnist no shearing
models = [Setup.GEN[0], Setup.GEN[1], Setup.GEN[3], Setup.GEN[4]]
#legends label
labels= ["MNIST", "affNIST", "my MNIST", "my affNIST", "my affNIST w/o shear"]
#model version
model_version = args.model_version
#colors
colors = list(mcolors.BASE_COLORS.keys())

figure, axis = plt.subplots(2, 2, figsize=(9, 4.5), sharex=True)
axis = axis.flatten()

for id in range(0, 4):
    if id == 3: #wait for no sharing model to be complete
        continue

    file_path = "./"+models[id]+model_version+".txt"
    result = parse(file_path)
    no_epochs = len(result)
    # x axis values
    x = [*range(1, no_epochs+1)]
    #train
    y = [epoch["Train"] for epoch in result]
    axis[id].plot(x, y, label = "Train", color=colors[0])

    for i in range(0, len(Setup.GEN)):
        if i == 2 :
            continue

        if args.exclude_affnist and i == 1 and id != 1:
            continue

        y = [epoch[Setup.GEN[i]] for epoch in result]
        axis[id].plot(x, y, label = labels[i], color=colors[i+1])
        axis[id].title.set_text("Model: "+models[id])

h, l = axis[1].get_legend_handles_labels()

plt.legend(h, l, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75, wspace=0.2)
plt.show()