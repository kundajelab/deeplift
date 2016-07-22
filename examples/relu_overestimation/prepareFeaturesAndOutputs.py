#!/usr/bin/env python
from __future__ import division;
from __future__ import print_function;
from __future__ import absolute_import;
import sys;
import os;
scriptsDir = os.environ.get("UTIL_SCRIPTS_DIR");
if (scriptsDir is None):
    raise Exception("Please set environment variable UTIL_SCRIPTS_DIR");
sys.path.insert(0,scriptsDir);
import pathSetter;
import util;
import fileProcessing as fp;
import numpy as np

def prepareFeatureAndLabels(options):

    #prepare features and outputs files
    features_fh = fp.getFileHandle("features.gz", 'w')
    features_fh.write("id\t"+
      "\t".join(["inp"+str(x) for x in range(options.num_inputs)])+"\n")
    outputs_fh = fp.getFileHandle("outputs.gz", 'w') 
    outputs_fh.write("id\toutput\n")

    thresholds = np.arange(0.0, 1.0, 1.0/options.num_inputs)
    total_examples_generated = 0
    while total_examples_generated < options.num_examples:
        if (total_examples_generated%1000 == 0):
            print("total examples:",total_examples_generated)
        features = np.random.random((options.num_inputs,)) 
        features = np.round(features, decimals=2)
        output = np.sum(np.maximum(features-thresholds,0.0))
        features_fh.write("example"+str(total_examples_generated)+"\t"
                          +"\t".join("%0.2f"%float(x) for x in features)+"\n")
        outputs_fh.write("example"+str(total_examples_generated)+"\t"
                        +str(float(output))+"\n") 
        total_examples_generated += 1
    features_fh.close()
    outputs_fh.close() 

if __name__ == "__main__":
    import argparse;
    parser = argparse.ArgumentParser();
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--num_inputs", type=int, required=True)
    options = parser.parse_args();
    prepareFeatureAndLabels(options)
