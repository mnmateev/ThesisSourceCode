# thesis

train.py - the data & model are loaded. then the model is compiled, trained, and predicts in this file.
           VAE model is used to help keep track of changes over different measures - more helpful than convolution which mainly focuses
           on adjacent data points.

createData.py - a simple script I wrote to make data creation easier.
                it converts MIDI files into numpy arrays for the model.

predictionMethods.py - methods which parse the model's output into a MIDI file.

Libraries required:

- numpy
- pandas
- sklearn
- tensorflow/keras (all of the above for ML)
- mido (for reading and writing MIDI files)
