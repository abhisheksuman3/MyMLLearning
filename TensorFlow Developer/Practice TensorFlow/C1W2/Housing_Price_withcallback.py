import numpy as np
import tensorflow as tf

def create_training_data():
    """Creates the data that will be used for training the model.

    Returns:
        (numpy.ndarray, numpy.ndarray): Arrays that contain info about the number of bedrooms and price in hundreds of thousands for 6 houses.
    """
    
    ### START CODE HERE ###
    nx=np.array([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0]], dtype=float)
    ny=np.array([[0.5],[1.0],[1.5],[2.0],[2.5],[3.0]], dtype=float)+0.5
    # Define feature and target tensors with the values for houses with 1 up to 6 bedrooms. 
    # For this exercise, please arrange the values in ascending order (i.e. 1, 2, 3, and so on).
    # Hint: Remember to explictly set the dtype as float when defining the numpy arrays
    n_bedrooms = nx
    price_in_hundreds_of_thousands = ny

    ### END CODE HERE ###

    return n_bedrooms, price_in_hundreds_of_thousands




def define_and_compile_model():
    model=tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='sgd',loss='mse')
    return model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        '''
        Halts the training when the loss falls below 0.4

        Args:
            epoch (integer) - index of epoch (required but unused in the function definition below)
            logs (dict) - metric results from the training epoch
        '''

        # Check the loss
        if logs['loss'] < 0.4:

            # Stop if threshold is met
            print("\nLoss is lower than 0.4 so cancelling training!")
            self.model.stop_training = True

if __name__ == "__main__":
    features, targets = create_training_data()

    print(f"Features have shape: {features.shape}")
    print(f"Targets have shape: {targets.shape}")
    model = define_and_compile_model()

    model.summary()

    model.fit(features, targets, epochs=500,callbacks=[myCallback()])

    new_n_bedrooms = np.array([7.0])
    predicted_price = model.predict(new_n_bedrooms, verbose=False).item()
    print(f"Your model predicted a price of {predicted_price:.2f} hundreds of thousands of dollars for a {int(new_n_bedrooms.item())} bedrooms house")