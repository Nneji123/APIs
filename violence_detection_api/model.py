import tensorflow as tf
model_0 = tf.keras.models.load_model('bestmodel.h5')


def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    prediction = model_0.predict(inp)
    output  = ""
    if prediction[0][prediction.argmax()] < 0.84:
      output = "good image"
    elif prediction.argmax() == 0:
      output = "Rifle violence"
    elif prediction.argmax() == 1:
      output = "guns violence"
    elif prediction.argmax() == 2:
      output = "knife violence"
    elif prediction.argmax() == 3:
      output = "image porno"
    elif prediction.argmax() == 4:
      output = "personne habillée" 
    else:
      output = "tank violence" 
    return output
