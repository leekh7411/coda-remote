import os
import copy
import tempfile
import json
from subprocess import call
from diConstants import (HG19_ALL_CHROMS, MM9_ALL_CHROMS,
    HG19_TRAIN_CHROMS, MM9_TRAIN_CHROMS,
    VALID_CHROMS, TEST_CHROMS) 

import models
import modelTemplates

GM_MARKS = ['H3K27AC'] # Is this output mark?
#GM_MARKS = ['H3K27AC', 'H3K4ME1', 'H3K4ME3', 'H3K27ME3', 'H3K36ME3']
modelList = []
def init_test_GM18526():
    for test_cell_line in ['GM18526']:
        for subsample_target_string in ['0.5e6']:
            for predict_binary_output in [True, False]: # original : TRUE , FALSE   
                for output_mark in GM_MARKS:                            

                    model_params = modelTemplates.make_model_params(
                                #### Prediction model(keras ver) specification ####
                                model_library='keras',
                                model_class='SeqToSeq',
                                model_type='dcgan',
                                model_specific_params={
                                    'num_filters': 6,
                                    'filter_length': 251
                                },
                                compile_params={            
                                    'optimizer': 'adagrad'
                                },

                                #### Dataset specification ####
                                dataset_params={
                                    'train_dataset_name': 'GM12878_5+1marks-K4me3_all',
                                    'test_dataset_name': '%s_5+1marks-K4me3_all' % test_cell_line, #(1)
                                    'num_train_examples': 10000,
                                    'seq_length': 1001,
                                    'peak_fraction': 0.5,                            
                                    'train_X_subsample_target_string': subsample_target_string, #(2)
                                    'num_bins_to_test': None,
                                    'train_chroms': HG19_ALL_CHROMS, # GM-12878 ch1 ~ ch22
                                    'test_chroms': HG19_ALL_CHROMS, # CM-18526 ch1 ~ ch22
                                    'only_chr1': True
                                },
                                output_marks=[output_mark],
                                train_params={
                                    'nb_epoch': 10,
                                    'batch_size': 1000
                                },
                                predict_binary_output=predict_binary_output,
                                zero_out_non_bins=True,
                                generate_bigWig=True)

                    init_model(model_params)

def init_model(model_params):
    m = models.SeqModel.instantiate_model(model_params)
    modelList.append([m,model_params])
    print '\n\n\n'


    
init_test_GM18526()


def train_model():
    print 'Number of Model:', len(modelList)
    for model, params in modelList:
        print model
        model.train_model()
        
def evaluate_model():
    for model, params in modelList:
        print model.evaluate_model()


train_model()

#evaluate_model()    




























