import sys
sys.path.append("")
import os 
import tensorflow as tf

from tensorflow import keras


class LivenessNet:

	@staticmethod
	def build(width, height, depth, classes):
		
		data_augmentation = keras.Sequential(
			[
				tf.keras.layers.RandomFlip("horizontal", input_shape=(width, height, depth)), 
				tf.keras.layers.RandomRotation(0.1),
				tf.keras.layers.RandomZoom(0.1),
				tf.keras.layers.RandomTranslation(0.2, 0.2),
				tf.keras.layers.RandomBrightness(factor=0.2),
				# tf.keras.layers.RandomContrast(factor = 0.2),
				# tf.keras.layers.RandomCrop(height, width,)
			]
			)

		model = tf.keras.Sequential()
		model.add(data_augmentation)  # Correctly ,
		model.add(tf.keras.layers.Rescaling(1./255))
		model.add(tf.keras.applications.resnet50.ResNet50(
										include_top=False,
										weights='imagenet',
										input_tensor=None,
										pooling=None,
									))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(2048, activation='relu'))
		model.add(tf.keras.layers.Dropout(0.5))
		model.add(tf.keras.layers.Dense(512, activation='relu'))
		model.add(tf.keras.layers.Dropout(0.3))
		model.add(tf.keras.layers.Dense(64, activation='relu'))
		model.add(tf.keras.layers.Dense(classes, name='fc2'))

		# return the constructed network architecture

		return model

if __name__ == "__main__":


    width, height, depth, classes = 224, 224, 3, 2  # Change the parameters accordingly

    liveness_net_model = LivenessNet.build(width, height, depth, classes)
    liveness_net_model.summary()