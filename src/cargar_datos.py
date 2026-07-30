import os
import pandas as pd

def cargardatos():
    #obtiene la ruta del directorio donde se encuentra este script
    script_dir = os.path.dirname(__file__)
    #sube a la carpeta raiz
    project_dir = os.path.dirname(script_dir)
    #construye la ruta completa del archivo de base de datos
    file_path = os.path.join(project_dir, 'Base_de_datos.xlsx')
    #carga el archivo en un dataframe
    df = pd.read_excel(file_path)

    return df