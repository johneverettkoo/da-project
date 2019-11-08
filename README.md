# Data Analysis Project

Paper by S. Bambach, D. Crandall, L. Smith, C. Yu:
[Toddler-Inspired Visual Object Learning](http://vision.soic.indiana.edu/papers/diversity2018nips.pdf)

Complete:

* "Diverse" vs "similar" sampling demo
* Size experiment on Stanford Dogs dataset (in `misc`)
* Diversity experiment on Stanford Dogs dataset (in `misc`)
* Diversity experiment on CIFAR-10 dataset
* Diversity experiment on MNIST dataset

In progress:

* Empirical results on how the image sampling method corresponds to Inception Score
* A (slightly more) formal write-up of results thus far
* A more disciplined take on object size
* Tweak the hyperparameters
* Change the validation set from a random subset to something that also incorporates similarity/diversity
* Qualitatively selecting diverse/similar training sets

Notes:

* "Unclean" code is in `misc` (will clean up later)