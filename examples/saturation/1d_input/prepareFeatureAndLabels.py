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
    num_inputs = options.num_caps*options.inputs_per_cap 
    assert num_inputs%2 == 0
    num_tasks = int(options.num_caps*(options.num_caps-1)/2)
    total_examples_generated = 0 

    #prepare features and labels files
    features_fh = fp.getFileHandle("features.gz", 'w')
    features_fh.write("id\t"+
      "\t".join(["inp"+str(x) for x in range(num_inputs)])+"\n")
    labels_fh = fp.getFileHandle("labels.gz", 'w') 
    labels_fh.write("id\t"+
      "\t".join(["task"+str(x) for x in range(num_tasks)])+"\n")

    while total_examples_generated < options.num_examples:
        if (total_examples_generated%1000 == 0):
            print("total examples:",total_examples_generated)
        features = np.random.random((num_inputs,))-0.5 
        labels = np.zeros(num_tasks)
        total_positives = 0
        task_idx=0
        for cap1 in range(options.num_caps):
            cap1_met = (np.sum(features[(cap1*options.inputs_per_cap):
                                        ((cap1+1)*options.inputs_per_cap)])
                        > 0)
            for cap2 in range(cap1+1, options.num_caps):
                cap2_met = (np.sum(features[(cap2*options.inputs_per_cap):
                                        ((cap2+1)*options.inputs_per_cap)])
                            > 0)
                if (cap1_met and cap2_met):
                    labels[task_idx] = 1
                task_idx += 1 
        features_fh.write("example"+str(total_examples_generated)+"\t"
                          +"\t".join("%0.2f"%float(x) for x in features)+"\n")
        labels_fh.write("example"+str(total_examples_generated)+"\t"
                        +"\t".join(str(int(x)) for x in labels)+"\n") 
        total_examples_generated += 1
    features_fh.close()
    labels_fh.close() 

if __name__ == "__main__":
    import argparse;
    parser = argparse.ArgumentParser();
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--inputs_per_cap", type=int, required=True)
    parser.add_argument("--num_caps", type=int, required=True)
    options = parser.parse_args();
    prepareFeatureAndLabels(options)
