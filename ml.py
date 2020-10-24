from pickle import load
import tensorflow as tf
import numpy as np

def model_predict(usuario: dict):
    scaler = load(open('preprocessed_models/scaler.pkl', 'rb'))
    one_hot = load(open('preprocessed_models/one_hot.pkl', 'rb'))
    lb = load(open('preprocessed_models/lb.pkl', 'rb'))

    categorical_values = [
        usuario.genero,
        usuario.pais,
        usuario.posee_tarjeta,
        usuario.miembro_activo,
        usuario.velocidad_servicio
    ]

    numerical_values = [
        usuario.edad,
        usuario.antiguedad,
        usuario.facturacion,
        usuario.puntuacion_crediticia,
        usuario.cantidad_productos,
        usuario.salario_estimado,
    ]

    cat_one_hot = one_hot.transform(np.array(categorical_values).reshape(1,-1))
    num_scale = scaler.transform(np.array(numerical_values).reshape(1,-1))

    cat_one_hot = cat_one_hot.toarray()

    values = np.concatenate((cat_one_hot, num_scale), axis=1)

    model = tf.keras.models.load_model('models/', compile=True)

    pred = model.predict_classes(values, verbose=1)

    actual_label_prediction = lb.inverse_transform(pred)

    return actual_label_prediction[0]