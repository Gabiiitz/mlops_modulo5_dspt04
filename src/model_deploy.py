from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title='api de prediccion de pago de creditos')

MODEL_PATH = 'best_model.pkl'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class CustomerData(BaseModel):
    tipo_credito: int
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str
    anio_prestamo: int
    mes_prestamo: int

@app.get('/')
def home():
    """
    endpoint simple de bienvenida
    """
    return {'mensaje': 'api funcionando correctamente'}

@app.post('/predict')
def predict(data: CustomerData):
    """
    endpoint para recibir los datos de un cliente y retornar la prediccion
    """
    if model is None:
        raise HTTPException(status_code=500, detail='el modelo no esta cargado. entrene el modelo primero.')
    
    input_data = pd.DataFrame([data.dict()])
    
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    
    return {
        'pago_a_tiempo_prediccion': prediction,
        'probabilidad_pago': round(probability, 4)
    }