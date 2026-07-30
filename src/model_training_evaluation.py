import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import joblib

from ft_engineering import load_and_clean_data, build_model, summarize_classification

def train_and_evaluate():
    """
    esta funcion entrena los modelos, hace validacion cruzada, saca graficos y guarda el mejor modelo
    """
    # 1. cargar datos limpios
    X, y = load_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. definir los modelos
    models = {
        'Logistic Regression': build_model(LogisticRegression(max_iter=1000, random_state=42)),
        'Random Forest': build_model(RandomForestClassifier(n_estimators=100, random_state=42)),
        'XGBoost': build_model(XGBClassifier(eval_metric='logloss', random_state=42))
    }
    
    best_f1 = 0
    best_model_name = ''
    best_model_pipeline = None
    
    plt.figure(figsize=(15, 5))
    
    f1_results = {}
    
    for name, pipeline in models.items():
        print(f'\n--- entrenando y evaluando: {name} ---')
        
        # validacion cruzada (5 cortes)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        print(f'f1 score promedio en validacion cruzada: {cv_scores.mean():.4f}')
        
        # entrenar modelo
        pipeline.fit(X_train, y_train)
        
        # predicciones
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # resumen
        summarize_classification(y_test, y_pred)
        
        current_f1 = f1_score(y_test, y_pred)
        f1_results[name] = current_f1
        
        # guardamos si es el mejor
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_name = name
            best_model_pipeline = pipeline
            
        # grafico roc auc
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, label=f'{name} (auc = {roc_auc:.2f})')
        
        # grafico precision-recall
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.subplot(1, 3, 2)
        plt.plot(recall, precision, label=f'{name}')

    # formato grafico roc
    plt.subplot(1, 3, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('curva roc')
    plt.xlabel('tasa de falsos positivos')
    plt.ylabel('tasa de verdaderos positivos')
    plt.legend()
    plt.grid(True)

    # formato grafico precision-recall
    plt.subplot(1, 3, 2)
    plt.title('curva precision-recall')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend()
    plt.grid(True)

    # subgrafico 3: comparativa f1 score
    plt.subplot(1, 3, 3)
    plt.bar(f1_results.keys(), f1_results.values(), color=['skyblue', 'orange', 'green'])
    plt.title('comparativa de f1 score')
    plt.ylabel('f1 score')
    plt.ylim(0, 1)
    for i, v in enumerate(f1_results.values()):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig('evaluacion_modelos.png')
    plt.show()
    
    print('\n==========================================')
    print(f'el mejor modelo fue: {best_model_name} con f1 score = {best_f1:.4f}')
    print('==========================================')
    
    # guardar el mejor modelo
    joblib.dump(best_model_pipeline, 'best_model.pkl')
    print("modelo guardado exitosamente como 'best_model.pkl'")

if __name__ == '__main__':
    train_and_evaluate()