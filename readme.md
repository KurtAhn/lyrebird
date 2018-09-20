# Handwriting Generation Code Test

I implemented the networks for unconditional and conditional handwriting generation, but didn't attempt the optional ask of handwriting recognition. Models were implemented in Tensorflow and Python3 and can be found under `models/kurt.py`. Code used for data preparation can be found in `data/__init__.py`. To see how I trained the models, please refer to `train-unconditional.py` and `train-conditional.py`. The TensorFlow model definitions used for the model synthesis are under `mdldef/`.

Please note the following:
1. I was unable to implement random seeding, since setting the random seed after model creation in Tensorflow is apparently not at all straightforward.
2. Because seeding isn't supported, you may need to run the models a few times to see how well they perform.
3. Due to limited computational resources and time, I was only able to train the model for 4-5 epochs, so the results aren't great.
4. I made one slight change to the Jupyter notebook where I wasn't supposed to. In the bottom cell, I changed the `print` call to the Python3-compliant `print()`. I hope that's okay.

Thanks!
