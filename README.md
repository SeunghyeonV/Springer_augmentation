# Springer_augmentation

An empirical analysis of image augmentation against model inversion attack in federated learning - code used for augmentation experiment.

https://link.springer.com/article/10.1007/s10586-022-03596-1

Description

A defense strategy based on augmentation of image data against model inversion attack (An attack strategy that reconstructs hidden data that is not open to the public from shared gradient or trained parameters in federated learning or collaborative learning environment). By modifying the training data before the collaborative learning process, we proved that a simple augmentation can provide enhanced defense performance compared to a conventional differential privacy-based defense strategy (DP-SGD, an optimizer with injected noise).

For example, posterize augmentation (removing bits from RGB channels of image based on magnitude) with the highest magnitude can still provide meaningful training accuracy while successfully defending model inversion attacks.
