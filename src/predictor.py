import numpy as np

labels = [chr(i) for i in range(97, 123)]

def process(points):
    X = points.copy()
    mean = points.mean(axis=0)
    dist_norm = np.linalg.norm(X[0] - X[9])
    X = (X - mean) / dist_norm
    return X.reshape(63,)

def predict(model, points):
    X = process(points)
    prediction = model.predict([X])
    return labels[int(prediction[0])], np.max(model.predict_proba([X]))