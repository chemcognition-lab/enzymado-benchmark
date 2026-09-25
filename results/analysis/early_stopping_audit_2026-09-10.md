# Auditoría de early stopping — 2026-09-10

La afirmación de que todos los resultados actuales se entrenaron sin early stopping es falsa. El conjunto mezcla protocolos: 952 ejecuciones tienen evidencia de early stopping habilitado y 1.008 no lo usaron. Habilitado no significa necesariamente que el entrenamiento terminara antes del máximo.

## Alcance y método

Se revisaron los 1.960 directorios con hparams.json de results/predictions: 1.400 de properties y 560 de real_case. Se leyeron todos sus training_log.csv. Se buscaron 13.481 líneas de guardado en logs/*.out recursivamente y se emparejó cada artefacto por modelo, embedding, tarea, split y pooling. Se normalizó únicamente el nombre histórico loco a similarity. Entre coincidencias repetidas se escogió la de menor distancia entre la fecha de modificación del log y la del historial. Los 1.960 tienen una coincidencia a menos de 24 horas, repartida entre 743 archivos de log. Los historiales datan del 14–16 y 20–21 de agosto de 2026.

Los 1.568 historiales neuronales concuerdan con el bloque de entrenamiento correspondiente: 896 con best_val_loss y validación en el log, 672 sin ambos. También se inspeccionaron los metadatos binarios de los 84 model.joblib XGBoost con early_stopping_rounds configurado, sin deserializar objetos ni ejecutar entrenamiento. El CSV adjunto conserva una fila por resultado, la clasificación, las épocas, el log y las líneas de inicio y guardado.

## Resultado por grupo

| Grupo | Ejecuciones | Evidencia |
|---|---:|---|
| Attention, Linear y MLP, ambas categorías | 784 | Early stopping habilitado, con mejor época y pérdida de validación |
| No attention, Linear y MLP, active_sites y binding_site | 112 | Early stopping habilitado, con mejor época y pérdida de validación |
| No attention, Linear y MLP, otras tareas | 672 | Sin validación efectiva; 200/200 épocas Linear y 100/100 MLP; best_epoch y best_val_loss vacíos |
| XGBoost estándar: regresión, binding y solubility | 308 | Sin early_stopping_rounds en la configuración |
| XGBoost ec_number | 28 | Paciencia 20 configurada, pero sin validación efectiva; todos los boosters guardados tienen 50 árboles y carecen de best_iteration/best_score |
| XGBoost active_sites y binding_site | 56 | Paciencia 10 habilitada; todos tienen best_iteration=39 y 40 árboles: no acortaron el entrenamiento |

De las 896 ejecuciones neuronales con early stopping habilitado, 877 terminaron por paciencia antes del máximo y 19 alcanzaron el máximo. En dos de estas últimas, la paciencia se cumplió justo en la última época. Completar todas las épocas no demuestra por sí solo ausencia de early stopping.

## Evidencia concreta de los resultados actuales

- Sin early stopping: logs/orig42_ph_opt_na_4742671_0.out, líneas 59–106. Para ph_opt/ankh3_large/random, anuncia Validation enabled: n_val=1,144, pero MLP entrena hasta 100/100 y Linear hasta 200/200 sin val_loss. Las líneas 75 y 101 guardan exactamente las rutas actuales. Sus training_log.csv contienen best_epoch y best_val_loss vacíos en todas las épocas.
- Con early stopping: logs/orig42_ph_opt_at_4742672_0.out, líneas 52–77. El MLP para la misma tarea, embedding y split se detiene en época 34, mejor época 14, best_val_loss=0.683642. Linear se detiene en 52, mejor época 32, best_val_loss=0.854777. Los historiales actuales coinciden.
- Excepción no_attention con validación: logs/orig42_active_na_4742655_0.out, líneas 55–76. MLP se detiene en 64 y restaura 44; Linear se detiene en 12 y restaura 7.
- XGBoost de residuos: el mismo log, líneas 81–92, construye matrices de entrenamiento y validación y registra best_iteration=39 y best_score=0.0148661303306753. El CSV vacío no significa que no hubiese validación.
- Un resultado XGBoost fue guardado desde un checkpoint ya completado: logs/o42l_active_xgb_resume_4806263_0.out, líneas 46–56. Su modelo también conserva best_iteration=39. Los otros 55 tienen el resumen de fit explícito en el bloque de log asociado.
- Real case sin atención: logs/rc42_noattn_4924173_0.out, líneas de guardado 72 y 98, para align_amylase_activity/ankh3_large. Las dos redes completan su presupuesto sin best_val_loss; los correspondientes modelos con atención sí registran validación.

## Causa histórica reproducida

En el commit 33570328, anterior a estas ejecuciones:

1. src/training/train.py::_fit_arrays envía x_val, en minúscula.
2. src/training/model_registry.py::train_model declara X_val, en mayúscula. El argumento x_val entra en extra_fit_kwargs, mientras X_val queda en None.
3. Los wrappers pooled también esperan X_val; el filtro de firma descarta x_val y no transmite y_val porque X_val es None.
4. El wrapper no construye val_payload; src/models/common.py ejecuta la rama sin validación, que devuelve None, None y completa todas las épocas.

Se ejecutó únicamente la función histórica extraída mediante AST, con un objeto de prueba que captura argumentos: enviar x_val=[2], y_val=[2] produjo X_val=None, y_val=None en fit. No se entrenó ningún modelo. La ruta con DataLoaders usa val_dataloader y evita este problema, lo que explica exactamente la separación observada.

El código actual usa x_val de manera consistente en esta ruta neuronal. Eso no modifica los resultados históricos. La reproducción demuestra el defecto en la versión histórica inspeccionada y su concordancia con los artefactos; no existe un run_manifest.json en estos 1.960 directorios que certifique el commit exacto de cada ejecución.

## Por qué los CSV de XGBoost pueden inducir a error

La versión registrada de write_training_log_file consulta best_iteration/best_score, pero los wrappers especializados guardan best_iteration_/best_score_. Hay una corrección de esos nombres ya presente en los cambios locales, anterior a esta auditoría. Por eso no se clasificaron estos modelos usando solamente las celdas vacías del CSV.

Los 28 modelos EC fueron inspeccionados directamente: 14 contienen 23 boosters y 14 contienen 24; todos los boosters tienen atributos vacíos y 50 árboles. Ninguno contiene best_iteration ni best_score, aunque el wrapper guarda early_stopping_rounds=20. Esto respalda la ausencia de validación y parada en esta ruta. En cambio, los 56 modelos de residuos guardan best_iteration=39, best_score y 40 árboles.

## Implicación y límites

Los resultados actuales comparan modelos entrenados con criterios de parada distintos. No se puede describir todo el benchmark como entrenado con early stopping ni como entrenado sin él. Si se exige uniformidad con early stopping, hay 700 ejecuciones con paciencia configurada pero validación perdida (672 neuronales y 28 EC XGBoost); los otros 308 XGBoost requieren además una decisión explícita de protocolo porque no lo tenían configurado. Reentrenar puede cambiar métricas y rankings, en una dirección que esta auditoría no permite predecir.

La vinculación a logs usa identificadores, líneas de guardado, concordancia de historial y proximidad temporal; no hashes de procedencia. No se recalcularon predicciones, métricas o ratings, ni se alteraron modelos, configuraciones o código. Se añadieron únicamente este informe y su CSV de evidencia.
