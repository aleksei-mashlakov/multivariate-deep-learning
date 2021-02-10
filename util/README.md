# Neural Network Utilities

This is a repository that contains generic utilities used by other deep learning projects.  
The following are currently available:
## model_util.py
A collection of generic functions that can be used to handle Tensorflow Keras models.
### Functionality
#### *`SaveModel(model, filename)`*
A function to save a model and its weights as follows:
* Model saved in file `filename.json`
* Weights saved in file `filename.h5`
##### Arguments
* __model:__ Keras model to be saved
* __filename:__ The full path of the filename without file extension.

#### *`LoadModel(filename, custom_objects)`*
A function to load a model and its weights as follows:
* Model read from file `filename.json`
* Weights read from file `filename.h5`
##### Arguments:
* __filename:__ The full path of the fulename without file extension
* __custom_objects:__ A dictionary of Keras custom layers (if any) that were defined for the model that is being read

#### *`SaveResults(model, init, history, test_results, metrics)`*
A function to save training, validation and testing results into a file.  
The saved results will be as follows:
* Training info: information about the training parameters used:
  - Loss function
  - Optimisation method
  - Learning rate
  - Batch size
  - Number of epochs
* Training results: The last value reached while training for each of the metrics that are available in history. The metrics to be printed are the ones passed in 'metrics' (an array of metrics)
* Validation results: Same as Training results (if available)
* Testing results: The same metrics values for testing (if available), passed via the dictionary 'test_result'
* Model Summary
##### Arguments
* __model:__ The Keras model
* __init:__ An object that must contain the following elements:
  - init.loss: Loss function that was used
  - init.optimiser: Optimiser used for training
  - init.lr: learning rate
  - init.batchsize: batch size (as its name indicates)
  - init.epochs: number of epochs
  - init.save: full path where the results will be saved. The filename without extension. a ".txt" will be added to this filename
  - init.validate: Boolean that indicates if validation on the validation set was done and hence results are available
  - init.evaltest: Boolean that indicates if validation on the test set was done and hence results are available
* __history:__ The history element from Keras .fit return object which is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable)
* __test_results:__ A list containing the metrics values for the test set evaluation (if applicable)
* __metrics:__ A list of the metrics against which the model was trained.


#### *`SaveHistory(filename, history)`*
A function to save all history values in order to be plotted later if needed  
The file will contain a column for each metric and as many records as the epochs values.  
File dimensions: (Epochs + 1 lines) x (Number of Metrics columns) ... where (+1 for the name of metrics)
The format for saving those is then as follows:
* First line will contain the metrics names separated by commas
* The next epochs-lines (one record for each epoch) will contain the values for each metric, also comma separated
##### Arguments
* __filename:__ The full path of the filename without file extension. A "\_history.csv" will be added to the filename
* __history:__ The history element from Keras .fit return object which is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable)


## Msglog.py
A module that optimises the *logging object* in order to generate logs with the following format:<br>
`YYYY-mm-dd HH:MM:SS.msec (process_id) (levelname) module.function -> message`<br>
Example:<br>
```2019-02-21 18:44:37.548 (668456) (INFO) main.<module> -> Tensorflow version: 1.11.0```<br>
This module will do the following:
- All messages with level Error or CRITICAL will be written on the terminal where the main python file is being run regardless of whether you want to generate logs on not. All error and critical messages are written on the terminal.
- If you choose to create logs, a logfile will be created with all messages whose level is greater than debuglevel.
### Functionality
#### *`LogInit(name, filename, debuglevel = logging.INFO, log = True)`*
Initialises logging. This function must be called in the main python file
##### Arguments
- __name:__ Name of the logger that is passed from the main python file and which must be used by all other files
- __filename:__ Full path of the file where logs are to be written. This is a *TimedRotatingFileHandler* that rotates at midnight.
- __debuglevel:__ The minimum log level that will be written into the file. All messages with level less than debuglevel will not be written into the file handler. Please check the logging class documentation for more info.
- __log:__ Whether to generate logs or not.

### Usage
In the main python file, do the following:
- Define a variable `logger_name` giving it the value that you want. This variable __must__ be defined at the beginning of the main python script before importing all other python files that uses the logging. Example: `logger_name = "Test"`
- `from Msglog import LogInit`
- `log = LogInit(logger_name, logfilename, debuglevel)`

In all other python files, add the following:
- `from __main__ import logger_name`
- `import logging`
- `log = logging.getLogger(logger_name)`

You can then write to logfile using the log class.<br>
Example: `log.info("Tensorflow version: %s", tf.__version__)`
