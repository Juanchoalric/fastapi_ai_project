from fastapi import FastAPI
from pydantic import BaseModel
from ml import model_predict 


app = FastAPI()


class Usuario(BaseModel):
    pais: str
    genero: str
    edad: int
    antiguedad: int
    facturacion: float
    puntuacion_crediticia: int
    cantidad_productos: int
    posee_tarjeta: str
    miembro_activo: str
    salario_estimado: float
    velocidad_servicio: str


@app.post('/abandono/')
def analize_article(usuario: Usuario):
    pred = model_predict(usuario) 
    return {"result": pred}