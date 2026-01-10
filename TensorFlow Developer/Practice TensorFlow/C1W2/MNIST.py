import os
import base64
import tensorflow as tf
import numpy as np

# Get current working directory
current_dir = os.getcwd()

print(f"Current directory: {current_dir}")
# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "mnist.npz")
print(f"data_path: {data_path}")

# Load from local mnist.npz if present, otherwise download via Keras
#with np.load(data_path, allow_pickle=True) as f:
#    training_images, training_labels = f['x_train'], f['y_train']
#        # test_images, test_labels = f['x_test'], f['y_test']  # optional

zipdata=np.load(data_path)   
test_images=(zipdata['x_test'])
test_labels=(zipdata['y_test'])
training_images=(zipdata['x_train'])
training_labels=(zipdata['y_train'])

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")
print(training_images)

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([ 
		tf.keras.Input(shape=(28,28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ]) 

    ### END CODE HERE ###
    
    # Compile the model
model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self, epoch, logs=None):
        
        # Check if the accuracy is greater or equal to 0.98
        if logs.get('accuracy') >= 0.98:
                            
            # Stop training once the above condition is met
            self.model.stop_training = True

            print("\nReached 98% accuracy so cancelling training!") 


fitmodel = model.fit(training_images, training_labels, epochs=10,callbacks=[EarlyStoppingCallback()])

predicted_label = model.predict(test_images, verbose=False)
predicted_label 