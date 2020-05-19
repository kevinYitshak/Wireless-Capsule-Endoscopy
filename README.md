# Wireless-Capsule-Endoscopy

This repo contains pytorch models for the segmentation
of nine different abnomalities based on kid-1 and as
well as on kid-2.

The data augmentation techniques are used with the
help of an external library called 'Alubmentation'
which uses cv2 rather than PIL, for faster
augmentation.

For better initialization for training the models,
I used an another library called smp segmentation models
pytorch.
