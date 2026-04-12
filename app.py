from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import tensorflow as tf
import numpy as np
import pickle
import os

app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/malaria_lstm_g6'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class RegistroPrediccion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fecha_consulta = db.Column(db.DateTime, default=datetime.now)
    anho = db.Column(db.Integer)
    semana_actual = db.Column(db.Integer)
    provincia = db.Column(db.String(50))
    s6 = db.Column(db.Float) # Usamos Float por si hay decimales, o Integer
    s5 = db.Column(db.Float)
    s4 = db.Column(db.Float)
    s3 = db.Column(db.Float)
    s2 = db.Column(db.Float)
    s1 = db.Column(db.Float)
    prediccion = db.Column(db.Float)

with app.app_context():
    db.drop_all() 
    db.create_all() 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelos', 'lstm_loreto_global.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'modelos', 'scaler_casos_loreto.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'modelos', 'encoder_provincias.pkl')

modelo = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f: encoder = pickle.load(f)

@app.route('/')
def index():
    # Mostramos los últimos 8 registros en el historial
    historial = RegistroPrediccion.query.order_by(RegistroPrediccion.fecha_consulta.desc()).limit(8).all()
    provincias = list(encoder.classes_)
    return render_template('index.html', provincias=provincias, historial=historial)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        prov_nombre = data['provincia']
        anho_actual = int(data['anho'])
        sem_a_predecir = int(data['semana_actual']) # La semana que eligió el usuario
        casos_raw = [float(x) for x in data['casos']]

        # 1. PROCESAMIENTO IA (Lógica de 1 paso)
        id_p = encoder.transform([prov_nombre])[0]
        v_ohe = np.zeros(len(encoder.classes_))
        v_ohe[id_p] = 1

        casos_input = np.array(casos_raw, dtype=np.float32).reshape(-1, 1)
        casos_esc = scaler.transform(casos_input).flatten()
        ventana = np.array([np.concatenate(([c], v_ohe)) for c in casos_esc])
        
        pred_raw = modelo.predict(ventana[np.newaxis, ...], verbose=0)[0, 0]
        res_final = int(round(scaler.inverse_transform([[pred_raw]])[0, 0]))

        # 2. GUARDAR EN MYSQL (Nutrición de datos con 7 columnas)
        # Aquí separamos la lista 'casos_raw' en columnas individuales s1 a s6
        nuevo = RegistroPrediccion(
            anho=anho_actual,
            semana_actual=sem_a_predecir,
            provincia=prov_nombre,
            s6=casos_raw[0], # La más antigua
            s5=casos_raw[1],
            s4=casos_raw[2],
            s3=casos_raw[3],
            s2=casos_raw[4],
            s1=casos_raw[5], # La más reciente (semana previa)
            prediccion=res_final # El resultado de la IA (semana actual)
        )
        db.session.add(nuevo)
        db.session.commit()

        return jsonify({'status': 'success', 'prediccion': res_final})
    except Exception as e:
        print(f"Error detectado: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/historial')
def obtener_historial():
    # Consultamos los últimos 10 registros
    registros = RegistroPrediccion.query.order_by(RegistroPrediccion.fecha_consulta.desc()).limit(10).all()
    
    output = []
    for r in registros:
        output.append({
            'fecha_consulta': r.fecha_consulta.isoformat(),
            'anho': r.anho,
            'semana_actual': r.semana_actual,
            'provincia': r.provincia,
            's6': r.s6, 's5': r.s5, 's4': r.s4,
            's3': r.s3, 's2': r.s2, 's1': r.s1,
            'prediccion': r.prediccion
        })
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)