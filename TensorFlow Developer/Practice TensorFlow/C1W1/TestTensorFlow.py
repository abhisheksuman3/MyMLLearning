import tensorflow as tf
import numpy as np

#xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
xs = np.array([[-1.0,  1.0], [1.0, 2.0], [3.0, 7.0],[4.0,  9.0], [7.0, 9.5], [8.0, 10.0]], dtype=float)
print(xs.shape)
print(xs[:,0])


ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Build a simple Sequential model
model = tf.keras.Sequential([

    # Define the input shape
    tf.keras.Input(shape=(2,)),
    #tf.keras.Input(shape=(1,)),

    # Add a Dense layer
    tf.keras.layers.Dense(units=1)
    ])


model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=10)

print(f"model predicted: {model.predict(np.array([[3.0,8.4]]), verbose=0).item():.5f}")
print(f"model predicted: {model.predict(np.array([[7.0,9]]), verbose=0).item():.5f}")

model.summary()
#print(f"mondel predicted: {model.predict(np.array([1.0]), verbose=0).item():.5f}")

'''
npr=model.predict(xs, verbose=0).item()
print(npr)
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(7, 5)) # Sets the size of the plot
plt.scatter(xs[:,0], ys,colorizer='blue')
plt.scatter(xs[:,1], ys,colorizer='red')
plt.scatter(npr, ys,colorizer='green')
plt.grid(True, linestyle=':', alpha=0.6) # Adds a subtle grid
plt.show()
'''


