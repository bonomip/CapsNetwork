# importing the required module
import re
import sys
import numpy as np
import argparse
sys.path.append("./src")
from setup import Setup # set up model and dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

'''

the files must be in "./MAIN_MODEL_VERSION" folder.

for example if using v2 "./v2"
for example if using v3 "./v3"


'''

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
parser.add_argument("--basic",
                    default=False,
                    action='store_true')
parser.add_argument("--model-version",
                    default="_v1",
                    type=str)
parser.add_argument("--main_model-version",
                    default="v2",
                    type=str)
args=parser.parse_args()    


#models to plot
            #MNIST       #affNIST
models = [Setup.GEN[0], Setup.GEN[1]]
#legends label
labels = ["Train", "Test"]    
#model version
model_version = args.model_version
main_model_version = args.main_model_version
#colors
colors = list(mcolors.BASE_COLORS.keys())

figure, axis = plt.subplots(2, 1, figsize=(5, 4.5))
axis = axis.flatten()

for id in range(0, len(models)):
    file_path = "./results/"+main_model_version+"/"+models[id]+model_version+".txt"
    result = parse(file_path)
    no_epochs = len(result)

    # x axis values
    x = [*range(1, no_epochs+1)]
    #train
    y = [epoch["Train"] for epoch in result]
    axis[id].plot(x, y, label = labels[0], color=colors[0])

    y = [epoch[models[id]] for epoch in result]
    axis[id].plot(x, y, label = labels[1], color=colors[1])


    axis[id].grid()

ticks = []

#mnist A
axis[0].title.set_text("Model A (MNIST 40x40)")
axis[0].set_yticks(np.arange(0.96, 1, 0.01))
axis[0].set_ylim(ymin=0.96, ymax=1)
ticks = np.arange(0, 36, 5)
ticks[0] = 1
axis[0].set_xticks(ticks)
axis[0].set_xlim(left=1, right=35)
#affnist B
axis[1].title.set_text("Model B (affNIST)")
axis[1].set_yticks(np.arange(0.88, 1, 0.02))
axis[1].set_ylim(ymin=0.88, ymax=1)
ticks = np.arange(0, 36, 5)
ticks[0] = 1
axis[1].set_xticks(ticks)
axis[1].set_xlim(left=1, right=33)
# mnist with 2pixel shift 28x28 C

# mnist + random pos 40x40

h, l = axis[1].get_legend_handles_labels()
plt.legend(h, l, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75, wspace=0.2)
figure.tight_layout(pad=1.0)
plt.show()