import tensorflow as tf                 # Porfavor ejecutar antes de cotinuar: pip install image
import h5py

f = h5py.File('ModeloV3.h5', 'r')
print("Modelo 3: ", list(f.keys()))


fileName = "ModeloV2.h5"
f = h5py.File(fileName,  "r")
for item in f.attrs.keys():
    print(item + ":", f.attrs[item])
""" mr = f['/entry/mr_scan/mr']
i00 = f['/entry/mr_scan/I00']
print("%s\t%s\t%s" % ("#", "mr", "I00"))
for i in range(len(mr)):
    print("%d\t%g\t%d" % (i, mr[i], i00[i])) """
f.close()

""" g = h5py.File('ModeloV2.h5', 'r')
print("Modelo 2: ", list(g.keys()))

h = h5py.File('ModeloV1.h5', 'r')
print("Modelo 1: ", list(h.keys()))

i = f.get('/model_weights')
i.items()
print("Pesos:", i.items()) """

""" model = tf.keras.models.load_model("mobilenet.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model) 
"""