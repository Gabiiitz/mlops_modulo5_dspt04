import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from cargar_datos import cargardatos

def clasificar_tipo_tendencias(x):
    """
    esta funcion estandariza los valores numericos de tendencia_ingresos
    a categorias ordinales legibles
    """
    if isinstance(x, (int, float)):
        if pd.isna(x):
            return x
        elif x > 0:
            return 'Creciente'
        elif x < 0:
            return 'Decreciente'
        else:
            return 'Estable'
    return x

def summarize_classification(y_true, y_pred):
    """
    esta funcion recibe los valores reales y las predicciones,
    y te imprime un resumen basico con la matriz de confusion y las metricas
    """
    print('--- resumen de clasificacion ---')
    print('exactitud (accuracy):', accuracy_score(y_true, y_pred))
    print('\nmatriz de confusion:')
    print(confusion_matrix(y_true, y_pred))
    print('\nreporte detallado:')
    print(classification_report(y_true, y_pred))

def build_model(estimator):
    """
    esta funcion recibe un algoritmo/estimador y le pega
    el pipeline de preprocesamiento para devolver el pipeline completo
    """
    preprocessor = get_preprocessor()
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])
    return model_pipeline

def load_and_clean_data():
    """
    esta funcion llama a cargardatos(), borra la columna puntaje,
    estandariza tendencia_ingresos, extrae anio y mes de la fecha
    y separa el dataframe en X e y
    """
    df = cargardatos()
    
    # borramos puntaje porque afecta la predictibilidad
    if 'puntaje' in df.columns:
        df = df.drop(columns=['puntaje'])
        
    # estandarizamos tendencia_ingresos para convertir numeros a categorias
    if 'tendencia_ingresos' in df.columns:
        df['tendencia_ingresos'] = df['tendencia_ingresos'].apply(clasificar_tipo_tendencias)
        
    # procesamos la fecha extrayendo anio y mes
    if 'fecha_prestamo' in df.columns:
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'])
        df['anio_prestamo'] = df['fecha_prestamo'].dt.year
        df['mes_prestamo'] = df['fecha_prestamo'].dt.month
        df = df.drop(columns=['fecha_prestamo'])
        
    # separamos las variables (X) de la meta a predecir (y)
    X = df.drop(columns=['Pago_atiempo'])
    y = df['Pago_atiempo']
    
    return X, y

def get_preprocessor():
    """
    esta funcion arma el transformador de columnas especificando que hacer
    con los numeros, las categorias normales y las ordinales
    """
    numeric_features = [
        'capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente',
        'total_otros_prestamos', 'cuota_pactada', 'puntaje_datacredito',
        'cant_creditosvigentes', 'huella_consulta', 'saldo_mora', 'saldo_total',
        'saldo_principal', 'saldo_mora_codeudor', 'creditos_sectorFinanciero',
        'creditos_sectorCooperativo', 'creditos_sectorReal',
        'promedio_ingresos_datacredito', 'anio_prestamo', 'mes_prestamo'
    ]
    
    categorical_features = ['tipo_credito', 'tipo_laboral']
    
    ordinal_features = ['tendencia_ingresos']
    # incorporamos 'Sin Especificar' en la jerarquia de categorias
    ordinal_categories = [['Decreciente', 'Estable', 'Creciente']]
    
    # pipeline para datos numericos
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # pipeline para datos categoricos
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # pipeline para datos ordinales
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # juntamos todo en un solo transformador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer, ordinal_features)
        ]
    )
    
    return preprocessor

if __name__ == '__main__':
    X, y = load_and_clean_data()
    print('datos cargados con exito.')
    print('forma de X:', X.shape)
    print('forma de y:', y.shape)