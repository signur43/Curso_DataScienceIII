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


Implementaremos una validación cruzada (K-Fold Cross-Validation) para evaluar la robustez de la Regresión Logística y verificar si los resultados perfectos persisten en diferentes particiones de los datos de entrenamiento.

¡Gracias por ejecutar la validación cruzada! Como puedes ver, los resultados de la validación cruzada también muestran una precisión perfecta (1.00) en todos los folds, con una desviación estándar de 0.00.

Esto refuerza fuertemente la hipótesis de data leakage. Significa que, incluso cuando el modelo se entrena y evalúa en diferentes subconjuntos de tus datos (los folds de la validación cruzada), sigue logrando un rendimiento perfecto.

Esto es una clara indicación de que el modelo está aprendiendo una relación trivial entre las características (X_train_tfidf) y la variable objetivo (y_train) que no debería existir en un escenario real y, por lo tanto, no es un reflejo genuino de la capacidad predictiva del modelo.

Siguientes Pasos Recomendados:

Necesitamos investigar más a fondo cómo se generó la columna main_topic y cómo texto_reclamo_processed se relaciona con ella. Es muy probable que el main_topic se derive de texto_reclamo de una manera que introduzca directamente la respuesta en las características que el modelo utiliza para predecir.

Para inspeccionar la correlación entre las palabras clave y el main_topic, vamos a revisar los temas identificados por el modelo LDA. Esto nos ayudará a entender si hay palabras clave demasiado específicas que el modelo está utilizando para predecir perfectamente los temas, lo que sería un indicio de fuga de datos.

Los temas identificados por LDA, con sus palabras clave principales, confirman lo que sospechábamos. Cada tema tiene palabras clave muy distintivas y específicas. Esto es una fuerte señal de data leakage. El modelo de clasificación está logrando una precisión perfecta porque las características (palabras en el texto procesado) ya contienen la 'respuesta' sobre a qué tema pertenece un reclamo, en lugar de inferirlo de manera más general.

Para resumir las conclusiones y los siguientes pasos:

Para abordar la fuga de datos, primero identificaremos las palabras clave que causan el problema y las eliminaremos del texto preprocesado. Luego, re-entrenaremos el modelo y evaluaremos su rendimiento de nuevo.

¡Excelente! Los resultados del modelo re-entrenado son mucho más realistas y confirman nuestra hipótesis de fuga de datos.

Observa los cambios clave:

Reducción de la Precisión Total: La precisión general ha bajado del 1.00 perfecto a un 0.96.
Variabilidad en las Clases: Ahora, no todos los temas tienen una precisión, recall y F1-score de 1.00. Por ejemplo, el Tema 3 muestra una precisión del 0.80 y el Tema 4 un recall del 0.78, lo que es mucho más típico en problemas de clasificación.
Resultados de Validación Cruzada Realistas: La validación cruzada ahora nos da una precisión media de 0.9700 con una pequeña desviación estándar de 0.0015. Esto indica que el modelo es consistentemente bueno, pero ya no 'perfecto', en diferentes subconjuntos de datos de entrenamiento.
Estos nuevos resultados demuestran que el modelo ya no está 'memorizando' las respuestas a través de palabras clave obvias. Ahora, la clasificación se basa en patrones más generales en el texto, lo que lo hace un modelo mucho más confiable para predecir temas en reclamos nuevos o no vistos. La matriz de confusión también mostrará errores de clasificación, lo que es una señal saludable de que el modelo está aprendiendo de manera más compleja.

¡Hemos logrado mitigar el problema de la fuga de datos y ahora tenemos un modelo más robusto y creíble!
