""" 
MIT License

Copyright (c) 2020 Yoga Suhas Kuruba Manjunath

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

dataset_folder = './Dataset'
input_data_folder = './Extracted_Data'

#root_data = "Dataset-Unicauca-Version2-87Atts.csv"
root_data = "Unicauca-dataset-April-June-2019-Network-flows.csv"

models = './models'
logs = './logs'
graphs = './graphs'

x_shape = ()

epochs = 1000
batch_size = 1024

MAX_FLOWS_PER_CLASS = 20

train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

model_t = ["model_1", "model_2", "model_3", "sao"]
feature_set_t = ["1","2","3","4","5","all","top_7","test"]

FEATURE_SET_ALL = "all"
FEATURE_SET_TOP_7 = "top_7"
FEATURE_SET_1 = 1
FEATURE_SET_2 = 2
FEATURE_SET_3 = 3
FEATURE_SET_4 = 4
FEATURE_SET_5 = 5
FEATURE_SET_TEST = "test"

nb_of_features = 0

STATIC_NUM_CLASSES = 78 #currently we have 78 class

num_classes = 0
ntc_label = {}
