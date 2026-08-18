Descripción del Proyecto Este proyecto tiene como objetivo analizar y predecir el tiempo de respuesta y la satisfacción del cliente en el contexto de reclamos bancarios en Chile. Utilizamos un conjunto de datos que contiene, información detallada sobre los reclamos, incluyendo la institución bancaria, el producto, el motivo principal del reclamo, el canal de ingreso, la región del usuario, el monto disputado, el estado actual del reclamo y el tiempo de respuesta en días.

El análisis se enfoca en:

Exploración de Datos (EDA): Para entender la distribución de los reclamos, identificar tendencias y visualizar relaciones entre variables. Preparación de Datos: Incluyendo el manejo de valores nulos y la codificación de variables categóricas (One-Hot Encoding y Label Encoding) para preparar los datos para modelos de Machine Learning. Modelado Predictivo: Construcción de modelos de regresión (Regresión Lineal y Árbol de Decisión) para predecir el tiempo_respuesta_dias y modelos de clasificación (Regresión Logística y Random Forest) para predecir la satisfaccion del cliente. Identificación de Factores Clave: Determinar qué variables tienen mayor influencia en el tiempo de respuesta y la satisfacción del cliente. Ejecución del Cuaderno (Notebook) Para ejecutar este cuaderno y replicar los resultados, siga los siguientes pasos:

Cargar el Conjunto de Datos: El cuaderno comienza cargando los datos desde un archivo CSV alojado en GitHub. Asegúrese de tener acceso a Internet para que pandas.read_csv() pueda descargar el archivo correctamente.

github_csv_url = 'https://raw.githubusercontent.com/signur43/Curso_DataScienceIII/main/dataset_reclamos_bancarios.csv' df = pd.read_csv(github_csv_url) ```

Ejecutar Todas las Celdas: Se recomienda ejecutar todas las celdas del cuaderno en orden secuencial. Esto asegurará que todas las transformaciones de datos, codificaciones y entrenamientos de modelos se realicen correctamente.

En Google Colab, puede hacerlo yendo a Entorno de ejecución > Ejecutar todo. Inspección de Salidas: Revise las salidas de cada celda, incluyendo gráficos, tablas y métricas de evaluación del modelo, para comprender los resultados de cada etapa del análisis.

Dependencias Este cuaderno utiliza las siguientes bibliotecas de Python. Asegúrese de tenerlas instaladas en su entorno:

pandas, numpy, matplotlib, seaborn, sklearn (scikit-learn), shap, Si usa Google Colab, estas bibliotecas generalmente vienen preinstaladas. De lo contrario, puede instalarlas usando pip:

Dado que el modelo actual es de clasificación (clasifica dígitos del MNIST), las métricas de regresión no son apropiadas. Sin embargo, podemos profundizar en la evaluación del rendimiento del modelo con otras metodologías de clasificación. Generaré el código para mostrarte la matriz de confusión y un informe de clasificación completo, lo que te dará una visión más detallada de cómo el modelo está clasificando cada dígito.

¡Excelente! Ya hemos ejecutado el código para obtener una evaluación más detallada de nuestro modelo de clasificación. La visualización de la matriz de confusión y el informe de clasificación nos proporcionan una visión clara de cómo se desempeña el modelo.

Matriz de Confusión (visualizada arriba): La matriz de confusión muestra el número de predicciones correctas e incorrectas para cada clase. Los números en la diagonal representan las predicciones correctas (donde el valor real coincide con la predicción). Podemos observar visualmente que la mayoría de los valores se concentran en la diagonal, lo que indica un alto rendimiento.

Informe de Clasificación: Aquí tienes un resumen del rendimiento por cada dígito (clase):

Precisión: La proporción de identificaciones positivas correctas. Nuestro modelo tiene una precisión muy alta (mayor al 93% para todas las clases), lo que significa que cuando predice un dígito, es muy probable que sea correcto.
Recall (Sensibilidad): La proporción de positivos reales que se identificaron correctamente. El modelo también tiene un recall muy bueno, especialmente para dígitos como el '1' (99%) y el '3' (99%), lo que significa que detecta la mayoría de las instancias de cada dígito.
F1-Score: Es la media armónica de la precisión y el recall, proporcionando una medida equilibrada del rendimiento. Los F1-scores son consistentemente altos, la mayoría por encima del 96%.
Soporte: El número de ocurrencias reales de cada clase en el conjunto de datos de prueba.
Rendimiento General:

Accuracy general: 0.97 (97%), lo que confirma que el modelo predice correctamente el 97% de los casos en general.
Macro Average y Weighted Average: Indican un rendimiento consistentemente alto en todas las clases, tanto si se les da el mismo peso (macro) como si se pondera por su frecuencia (weighted).
En resumen, el modelo tiene un rendimiento excelente en la clasificación de los dígitos MNIST, con una alta precisión, recall y F1-score en todas las clases.
