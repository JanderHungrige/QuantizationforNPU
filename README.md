# QuantizationforNPU
We try to quantize a h5 model for the use with the NXP i.mx8 Plus NPU 

The model is the keras-VGGFace model :
https://github.com/rcmalli/keras-vggface

FILES:

quantizationforNPU.py loads a saved h5 model and fully quantizes it to int8. The h5 model can be found here :  https://mega.nz/file/ZN8nhYxC#N7iJ8LfACH2N34n3tPkz6a5we6Fog4utIgcHOLx53iQ

The converter.experimental_new_converter = True

and

converter.target_spec.supported_types = [tf.int8]

make sure that also the input layer is set to int8.

The 0002.jpeg is a sample image for the representative_dataset_gen(). The model was trained on 224x224 images. Actually the images needs to be normalized before, however to only get it running this should be sufficiant. 

