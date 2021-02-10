import os
import sys
import numpy as np
import tensorflow as tf

from datetime import datetime
from shutil import copyfile

from tensorflow.keras.models import model_from_json

# This test is done so that if logger_name is not set in __main__, the script runs anyway without generating logs
try:
    from __main__ import logger_name
    import logging
    log = logging.getLogger(logger_name)
except ImportError:
    pass

#
# A generic function to save the model and the weights as follows:
# - Model saved in the file 'filename.json' as a json file
# - The weights in the file 'filename.h5'
#
# The arguments that this function takes are:
# - model:     The Keras model to save
# - filename:  Full path of the filename where the model shall be saved without the file extension. Example: dir1/dir2/model
# 
def SaveModel(model, filename):
    if filename is not None:
        try:
            log.info("Saving model and weights ...")
        except NameError:
            pass

        folder = os.path.dirname(filename)
        
        # Check if the directory to save the file at exists. If not create it
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except IOError as err:
                try:
                    log.critical("Error while creating folder: %s ... Error[%d]: %s", folder, err.errno, err.strerror)
                except NameError:
                    print("Error while creating folder (" + str(folder) + "). Error[" + str(err.errno) + "]: " + str(err.strerror))
        
        # Backup model file if it exists
        file = filename + ".json"
        if os.path.exists(file):
            try:
                log.warning("File %s already exists. Backing it up to %s", file, filename + datetime.today().strftime('%Y%m%d') + ".json")
            except NameError:
                pass

            copyfile(file, filename + datetime.today().strftime('%Y%m%d') + ".json")

        # Save the model in json file
        with open(file, "w") as json_file:
            try:
                log.debug("Saving model into %s", file)
            except NameError:
                pass
            json_file.write(model.to_json())

        # Backup weights file if it exists
        file = filename + ".h5"
        if os.path.exists(file):
            try:
                log.warning("File %s already exists. Backing it up to %s", file, filename + datetime.today().strftime('%Y%m%d') + ".h5")
            except NameError:
                pass

            copyfile(file, filename + datetime.today().strftime('%Y%m%d') + ".h5")

        # Save weights in h5 file
        try:
            log.debug("Saving weights into %s", file)
        except NameError:
            pass
        model.save_weights(file)


#
# A generic function to load a model and its weights as follows:
# - Model read from json file 'filename.json'
# - Weights read from file 'filename.h5'
# filename being the full filename without the extension. Example: dir1/dir2/model
#
# The arguments that this function takes are:
# - filename:  Full path of the filename from where the model shall be saved. The filename is without the file extension. Example: dir1/dir2/model
# - custom_objects: A dictionary of user defined layers. For example, if you model has 2 custom layers "Layer1" and "Layer2", then custom_objects
#   is defined as follows:
#      custom_objects = {
#              'Layer1': Layer1,
#              'Layer2': Layer2
#      }
#
def LoadModel(filename, custom_objects):
    model = None

    if filename is not None:
        try:
            log.info("Loading model and weights ...")
        except NameError:
            pass

        # load and create model from json file
        file = filename + ".json"
        if os.path.exists(file):
            with open(file, "r") as json_file:
                try:
                    log.debug("Loading model from: %s", file)
                except NameError:
                    pass

                model = model_from_json(json_file.read(), custom_objects=custom_objects)
        
            # load weights from h5 file into new model
            file = filename + ".h5"
            if os.path.exists(file):
                try:
                    log.debug("Loading weights from: %s", file)
                except NameError:
                    pass

                model.load_weights(file)
            else:
                try:
                    log.critical("File %s does not exist", file)
                except NameError:
                    print("File %s does not exist" % (file))
        else:
            try:
                log.critical("File %s does not exist", file)
            except NameError:
                print("File %s does not exist" % (file))
    
    return model

#
# Function to save training, validation and testing results into a file.
# The saved results will be as follows:
# 1- Training info: information about the training parameters used:
#    - Loss function
#    - Optimisation method
#    - Learning rate
#    - Batch size
#    - Number of epochs
# 2- Training results: The last value reached while training for each of the metrics that are
#    available in history which is a Keras callback dictionary containing all training and
#    validation (if available) metrics
#    The metrics to be printed are the ones passed in 'metrics' (an array of metrics)
# 3- Validation results: Same as Training results (if available)
# 4- Testing results: The same metrics values for testing (if available), passed via the dictionary
#    'test_result'
# 5- Model Summary
#
def SaveResults(model, init, history, test_result, metrics):
    if init is not None and init.save is not None:
        try:
            log.info("Saving results ...")
        except NameError:
            pass

        file = init.save + ".txt"
        if os.path.exists(file):
            try:
                log.warning("File %s already exists. Backing it up to %s", file, init.save + datetime.today().strftime('%Y%m%d') + ".txt")
            except NameError:
                pass
            copyfile(file, init.save + datetime.today().strftime('%Y%m%d') + ".txt")
        
        # Save information about Training and Validation parameters, hyper-parameters and results
        with open(init.save + ".txt", "w") as f:
            try:
                log.debug("Saving system info")
            except NameError:
                pass
            f.write("System Info:\n")
            f.write("\tPython version: " + str(sys.version) + "\n")
            f.write("\tTensorFlow version: " + str(tf.__version__) + "\n")
            f.write("\tKeras version: " + str(tf.keras.__version__) + "\n")
            try:
                log.debug("Saving training info")
            except NameError:
                pass
            f.write("\nTraining Info:\n")
            f.write("\tLoss Function: " + str(init.loss) + "\n")
            f.write("\tOptimisation Method: " + str(init.optimiser) + "\n")
            f.write("\tLearning Rate: " + str(init.lr) + "\n")
            f.write("\tBatch Size: " + str(init.batchsize) + "\n")
            f.write("\tNumber of Epochs: " + str(init.epochs) + "\n")

            try:
                log.debug("Saving training results")
            except NameError:
                pass
            f.write("\nTraining Results:\n")
            for m in metrics:
                key = m
                f.write("\t" + str(m.title()) + ": " + str(history[key][-1]) + "\n")

            if init.validate is True:
                try:
                    log.debug("Saving validation results")
                except NameError:
                    pass
                f.write("\nValidation Results:\n")
                for m in metrics:
                    key = "val_" + m
                    f.write("\t" + str(m.title()) + ": " + str(history[key][-1]) + "\n")

            if init.evaltest is True and test_result is not None:
                try:
                    log.debug("Saving testing results")
                except NameError:
                    pass
                f.write("\nTesting Results:\n")
                for m in metrics:
                    f.write("\t" + str(m.title()) + ": " + str(test_result[m]) + "\n")

            try:
                log.debug("Saving model summary")
            except NameError:
                pass
            f.write("\nModel Summary:\n")
            model.summary(print_fn=lambda x: f.write('\t' + x + '\n'))


#
# Save all history values in order to be plotted later if needed
#
# The format for saving those is as follows:
#    First line will contain the metrics names separated by commas
#    The next epochs-lines (one record for each epoch) will contain the
#    values for each metric, also comma separated
# So as a summary, the file will contain a column for each metric and as 
# many records as the epochs values.
# File dimensions: Epochs + 1 lines (+1 for the name of metrics)
#                  Number of Metrics columns
#
def SaveHistory(filename, history):    
    if filename is not None:
        try:
            log.info(" Saving history ...")
        except NameError:
            pass

        key_list = []
        for key in history.keys():
            key_list.append(key)
        header = ','.join(key_list)
        h_array = np.concatenate(
                      (np.array(list(history.keys())).reshape(-1, 1).T,
                        np.array(list(history.values())).T),
                        axis=0)

        file = filename + "_history.csv"
        if os.path.exists(file):
            try:
                log.warning("File %s already exists. Backing it up to %s", file, filename + datetime.today().strftime('%Y%m%d') + "_history.csv")
            except NameError:
                pass
            copyfile(file, filename + datetime.today().strftime('%Y%m%d') + "_history.csv")

        try:
            log.debug("Saving history into %s", file)
        except NameError:
            pass
        np.savetxt(file, np.array(list(history.values())).T, header=header, delimiter=',')
