### Brief
The Deep Learning Model which is used to detect Braille Characters and map it to English Character.

### Steps involved 
1) Dataset is downloaded from Kaggle 
2) Image is preprocessed involving steps like grayscale conversion, blurring, canny and threshold
3) Segmentation of images and data augmentation is not done because of the nature of the dataset.
4) Images are trained into CNN Model using Tensorflow Keras.
5) Evaluation of the models is taken.

## Tools Used:
1) Pytorch
2) OpenCV
3) Numpy
4) Pillow
5) Ultralytics
6) Streamlit
7) GTTS