import sys
sys.path.append("./src")
from setup import Setup # set up model and dataset

setup = Setup(train_cfg=Setup.d_k[3],
              test_cfg=Setup.d_k[0],
              should_be_trained=True, 
              should_create_dataset=False,
              model_v="_V1",
              epochs=50,
              debug=False)