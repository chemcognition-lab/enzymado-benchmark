# Verificación del diagnóstico y alcance de reentrenamiento

Fecha: 2026-09-10. Revisión de los 24 archivos modificados por el otro agente, de los artefactos actuales y de la auditoría previa. No se modificaron sus cambios ni se lanzaron entrenamientos del benchmark.

## Dictamen

El diagnóstico contiene problemas reales, pero exagera algunas conclusiones y las correcciones no resuelven todo lo que afirman. Adoptar el conjunto actual de cambios implica repetir 1.904 de las 1.960 ejecuciones de entrenamiento si se quiere publicar un conjunto coherente con ese protocolo. No implica volver a obtener datos brutos ni calcular todos los embeddings. Los 56 entrenamientos EC con atención conservan criterio y datos; uno de ellos conserva el problema de selección de época 0.

No conviene lanzar el benchmark completo todavía: el código nuevo sigue permitiendo guardar época 0, admite predicciones no finitas durante la selección y no valida la compatibilidad del criterio al reanudar checkpoints.

## Hallazgos de la revisión de cambios

1. **La solución no garantiza guardar un modelo entrenado.** `src/enzymado/models/common.py:448` sigue tratando el modelo inicial como candidato; el cambio está en la señal, no en esa política. Se comprobó con un clasificador lineal de una sola ponderación sobre cuatro ejemplos: al aumentar el peso de 0,1 a 0,6 y 1,1, la BCE baja 0,621268 → 0,350385 → 0,196209, mientras AUPRC permanece en 1. El entrenamiento se detiene en época 2, devuelve best_epoch=0 y restaura peso 0,1. Es un contraejemplo reproducible a la garantía de que el cambio elimina ese resultado; no una estimación de su frecuencia real.
2. **El nuevo objetivo permite ocultar valores no finitos.** `metrics_registry.auprc_from_scores` elimina scores no finitos; `validation_objective` reemplaza métricas no finitas por el peor valor. Con etiquetas [0,1,0,1] y scores [NaN,0.9,0.1,0.8], el objetivo devuelve -1: AUPRC perfecta tras excluir el NaN. El chequeo de finitud del bucle ya no detecta ese fallo. Esto requiere atención antes de una ejecución costosa.
3. **No hay un contrato seguro para migrar checkpoints entre criterios.** `maybe_resume_torch_wrapper` copia el estado anterior y `_initialize_early_stopping` acepta initial_best_val_loss sin reevaluarlo. El atributo nuevo validation_metric_ no se comprueba y el cálculo utiliza el atributo de clase validation_metric. Un checkpoint viejo carga con sus pérdidas y épocas históricas bajo una clase cuyo criterio puede ser ahora AUPRC/Kendall. Además, los historiales siguen llamando best_val_loss a la métrica negada; el nuevo criterio no está en params de hparams.json. Usar directorios nuevos evita mezclar ejecuciones publicadas con este estado.
4. **La documentación de métricas contiene afirmaciones falsas o no demostradas.** AUPRC con scores constantes y ambas clases sí está definida: el código devuelve 0,5 con dos positivos de cuatro. Que un dataset sea grande no garantiza concordancia entre pérdida y ranking. RMSE tampoco equivale a Kendall; escoger una métrica cambia el objetivo de selección. La documentación no debería prometer concordancia con todo el leaderboard, que utiliza varias métricas.
5. **Los tests existentes pasan, pero no cubren estas garantías nuevas.** Los cambios en los dos canarios actualizan números esperados. Pasar esos canarios no demuestra que desaparezcan los modelos de época cero ni que la migración de checkpoints sea segura.

## Verificación de las afirmaciones del informe

### 129 modelos de época cero: confirmado, con matices

Son 129 de las 896 redes con early stopping registrado; todos son attention. Desglose completo:

| Tarea | Cantidad |
|---|---:|
| binding | 43 |
| kcat | 22 |
| dtm | 21 |
| ddg | 19 |
| solubility | 15 |
| align_glucosidase_expression | 5 |
| align_glucosidase_activity | 3 |
| ec_number | 1 |

La frase precisa es que entrenaron y después restauraron el estado inicial, no que nunca ejecutaran entrenamiento. En binding/ankh3_large/random/linear/attention se cargó el modelo publicado: current_epoch_=20, best_epoch_=0 y todos los pesos finales coinciden exactamente con best_model_state_. La lógica histórica y los logs explican la selección en los demás casos.

La explicación basada únicamente en BCE no explica los 129: 70 corresponden a regresión. La discrepancia entre métricas y pérdida es una hipótesis posible, pero los historiales publicados guardan el mejor valor acumulado de pérdida, no las métricas de cada época que permitirían demostrar que el ranking estaba mejorando. También podrían intervenir optimización, sobreajuste o diferencias de distribución.

La brecha media de rating de properties sí coincide: MLP attention queda 139,53 puntos debajo y Linear 165,35. No se ha demostrado cuánto de esa brecha causa la selección de época cero. Haría falta una comparación controlada; cambiar a la vez datasets, validación, selección y boosting no aísla esa causa. En real_case, Linear attention tiene un rating medio superior a Linear no_attention.

### XGBoost estándar sin early stopping: confirmado

308 resultados no tenían early_stopping_rounds. El cambio añade paciencia y eval_set; es una modificación del protocolo de entrenamiento. Hay además 28 XGBoost EC con paciencia configurada que perdían la validación, identificados en la auditoría previa.

### Logs XGBoost vacíos: confirmado, sin necesidad de reentrenar para corregir el registro

Los wrappers especializados usan best_iteration_/best_score_; los estimadores estándar utilizan otros nombres. El arreglo actual contempla ambos. Los 56 modelos de sitios guardados conservan best_iteration=39 y best_score aunque su CSV estuviese vacío. Reparar un registro o restaurar compatibilidad no obliga a volver a entrenar.

### Duplicados de sitios: confirmado en los archivos originales de Git LFS

- active_sites: 90.436 filas originales, 89.725 secuencias únicas; 711 filas redundantes y 15 secuencias con posiciones discrepantes.
- binding_site: 158.681 filas originales, 157.433 secuencias únicas; 1.248 filas redundantes y 252 secuencias con posiciones discrepantes.

Los archivos nuevos tienen una fila por secuencia. La unión de posiciones cambia etiquetas y ponderación de ejemplos, así que las 280 ejecuciones de esas dos tareas quedan afectadas. La unión es una decisión de curación del benchmark; la discrepancia no demuestra por sí sola qué anotación biológica era errónea.

### Split random sesgado por tamaño: confirmado; no es ausencia total de aleatoriedad

El algoritmo baraja con semilla y luego ordena grupos por peso para asignarlos mediante un algoritmo voraz. Es un split agrupado balanceado por filas, no un muestreo uniforme de proteínas. kcat actual:

| Split | Filas | Proteínas | Mediciones por proteína |
|---|---:|---:|---:|
| train | 18.521 | 6.916 | 2,678 |
| val | 2.315 | 138 | 16,775 |
| test | 2.315 | 123 | 18,821 |

No hay secuencias compartidas entre train y test en esta tarea. Declarar el diseño no requiere reentrenar. Cambiar las asignaciones sí lo requiere; los cambios revisados no modifican el algoritmo de splitting.

### Tres tareas real_case sin señal: afirmación demasiado fuerte

Un CI que cruza cero no prueba ausencia de señal. Se revisaron los 70 modelos de cada tarea:

| Tarea | CI de Kendall enteramente positivo | CI que cruza cero |
|---|---:|---:|
| glucosidase melting | 0 | 70 |
| glucosidase expression | 4 | 66 |
| glucosidase activity | 30 | 39 |
| galox | 36 | 34 |
| petase | 70 | 0 |

Melting tiene solo 11 unidades de test y ancho medio de CI de Kendall 0,994. Es evidencia de incertidumbre elevada. No justifica por sí sola eliminar su peso ni declarar las tres tareas sin señal. Ponderar por precisión cambia la pregunta del leaderboard y puede favorecer tareas más grandes; no es una reparación neutral.

Galox tiene 12 secuencias y Petase 19; el 100% de las secuencias de test están en train. Sus metadatos especifican splitting por puntos de medición dentro de proteína. Evalúan condiciones/puntos nuevos sobre proteínas conocidas; no generalización a proteínas no vistas. Si se quiere ese segundo objetivo hay que cambiar el split y entrenar nuevamente esas tareas.

### Fallback de mutaciones: el riesgo existe; 14 filas no verificadas aquí

MutationDeltaEmbeddingBuilder llama lookup_row, que busca primero la secuencia y luego permite resolver por accession. Si la secuencia mutante no está y el accession devuelve el wildtype, puede producir delta cero. El builder no lo sustituye explícitamente por wildtype: el fallback ocurre en la resolución del embedding. No se verificó independientemente en esta revisión que exactamente 14 casos queden excluidos para cada embedding por allowed_ids; no debe tratarse como garantía nueva.

### Modelos antiguos y tiempo: no son razones suficientes para rehacer todo

Se restauraron correctamente XGBFlatECBlockClassifier de ec_number y XGBResidueBinaryClassifier de active_sites mediante restore_fitted_wrapper. Este instala los alias de módulos antiguos. Un joblib.load aislado puede fallar, pero el repositorio ya tiene el mecanismo de restauración.

La suma de duration_seconds del CSV histórico sí da 1.048,98 horas, y el 97,89% corresponde a las tres tareas grandes. Sin embargo suma 2.313 bloques históricos con intentos, reanudaciones, fallos y cargas/evaluación. No es una previsión validada de una repetición limpia bajo los nuevos criterios. Dividir por 14 no incorpora desequilibrios entre tareas, colas ni asignaciones disponibles.

Las mejoras concretas de accuracy y AUPRC de la prueba de humo citada no se reprodujeron aquí; no se encontraron sus artefactos en los logs inspeccionados. No se usaron para aprobar el lanzamiento.

## Alcance si se adoptan todos los cambios actuales

| Motivo | Ejecuciones afectadas |
|---|---:|
| Nuevo criterio AUPRC/Kendall en redes pooled | 1.232 |
| Early stopping nuevo en XGBoost estándar | 308 |
| Restaurar validación perdida en ec_number sin atención (56 redes + 28 XGBoost) | 84 |
| Datasets nuevos de active_sites y binding_site | 280 |
| Total afectado | 1.904 |
| ec_number con atención sin cambio de datos/criterio | 56 |

El CSV rerun_scope_review_2026-09-10.csv identifica cada ejecución. Una de las 56 no afectadas conserva best_epoch=0: si la política nueva exige descartar toda época cero, también habrá que resolver ese caso. No basta con repetirlo con igual semilla y configuración.

Se pueden reutilizar embeddings para las secuencias que permanecen, datos brutos y datasets no modificados. Tras repetir modelos afectados, corresponde generar sus predicciones y recalcular métricas, bootstraps y ambos ratings. Cambios solo de reporte, compatibilidad, probabilidades inferidas desde pesos existentes o ponderación no requieren entrenamiento por sí mismos.

Para ejecuciones afectadas, no se puede asumir que el último checkpoint permita seleccionar retrospectivamente la mejor época con otra métrica: no se guardó la trayectoria completa de pesos. Hay que iniciar entrenamientos nuevos para aplicar coherentemente el nuevo protocolo. Relanzar en results/predictions sin overwrite omite resultados ya completos; reanudar checkpoints antiguos tampoco implementa una repetición limpia. El resultado anterior debe conservarse como versión histórica separada.

## Comprobaciones ejecutadas

- Ruff del repositorio completo: sin errores.
- Suite completa en L40S, job 5355376: 189 tests pasan, 67,14 segundos de pytest.
- Duplicados originales, AUPRC constante y restauración XGBoost: comprobaciones en el nodo del mismo job; las salidas se obtuvieron, y el paso compartido terminó por cierre de la asignación al concluir pytest.
- Comprobación independiente job 5355387, salida 0: caso de época cero con BCE decreciente, NaN excluido de AUPRC y comparación exacta de pesos de un resultado publicado.

No se realizaron cambios de código ni de datasets durante esta revisión. Se añadieron este informe y el CSV de alcance.
