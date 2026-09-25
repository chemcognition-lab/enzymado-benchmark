# Recursos para la corrida completa: auditoría histórica

Se parte del mapa de los 1.960 resultados actuales a 743 logs. Se consultaron 57 identificadores de arrays/jobs con sacct y se recuperaron 746 jobs con logs de esos envíos, incluyendo cuatro intentos fallidos (dos TIMEOUT y dos FAILED). La cohorte ampliada se usa para no omitir reanudaciones. El CSV histórico anterior se consultó aparte: 397 estados finales, incluidos 53 OUT_OF_MEMORY y 52 TIMEOUT. Son cohortes diferentes; no sumar ni extrapolar sus conteos como tasa de fallo de v3.

## Método

- Duración: ElapsedRaw de cada asignación Slurm. Se cuenta una vez por job aunque produzca varios modelos. Incluye carga, entrenamiento, predicción y serialización.
- Memoria: MaxRSS del step batch, convertido de KiB a GiB; no es la memoria solicitada. Es un máximo del step, no una medición separada por familia ni necesariamente el mínimo de RAM requerido.
- Los grupos combinados suman tiempos por embedding y split, pero toman el máximo de memoria. Sumar jobs separados añade sus cargas de arranque y no predice exactamente un job combinado.
- Las continuaciones se suman con el intento inicial. A/ankh3_xl/similarity cambió de asignación de memoria entre tramos, de modo que 21.30 h es referencia acumulada, no una corrida limpia equivalente.
- by_embedding.csv contiene 322 perfiles del esquema propuesto. heuristic_time_h aplica 1.30 × tiempo histórico acumulado + 15 minutos, redondeado a 1/2/3/6/12/24/48/72 horas. Es margen operativo explícito, no intervalo de confianza ni garantía bajo el nuevo protocolo.

| Grupo | Tiempo histórico acumulado por embedding/split (h) | MaxRSS observado (GiB) | RAM orientativa | Tiempo / división recomendada |
|---|---:|---:|---|---|
| A | 2.27–21.30 | 25.7–160.0 | 160G | 6–24 h; 48 h para ankh3_xl/similarity |
| A' | 1.94–11.76 | 17.3–160.0 | 160G | 3–24 h |
| B | 1.73–6.91 | 160.0–160.0 | 160G | 3–12 h |
| D | 2.19–7.71 | 152.3–160.0 | 160G | 6–12 h |
| C | 2.49–12.56 | 78.9–160.0 | 160G | Separar familias; hasta 12 h por familia como base |
| G1 | 1.93–18.20 | 47.2–256.0 | 256G | 3–24 h |
| G2 | 1.16–8.60 | 157.9–256.0 | 256G | 2–12 h |
| G3 | 4.13–13.24 | 128.7–256.0 | 256G | Separar binding linear y active_sites |
| H | 1.11–1.98 | 144.2–288.0 | 256G provisional; histórico reservado 288G | 3 h |
| E | 0.17–0.23 | 11.7–33.6 | 24–48G por embedding | 1 h |
| F | 1.65–5.52 | 13.7–35.8 | 32–64G por embedding | 3–12 h agrupado, o dividir por tarea |
| I | 0.38–1.17 | 10.3–26.7 | 32–48G por embedding | 2 h |

## Cambios frente a la propuesta

- **A:** Separar el outlier que consumió 16 h más 5.3 h de continuación.
- **A':** 12 h deja sin margen el máximo de 11.76 h.
- **B:** 24 h es excesivo como reserva uniforme según este historial.
- **D:** 3 h y 128G no reproducen la configuración exitosa histórica.
- **C:** Si se agrupan: perfiles de 6/12/24 h; máximo acumulado 12.56 h.
- **G1:** Mantener margen para el caso de 18.20 h.
- **G2:** Las reejecuciones fallidas se incluyen en el acumulado.
- **G3:** Agrupado: 6/12/24 h; máximo 13.24 h incluyendo continuación.
- **H:** El nuevo manifest mean-pooled cambia la carga: no inferir un mínimo de 288G. No hay medición equivalente v3 para recortar con confianza.
- **E:** Referencia histórica 10–14 min; ES y nueva métrica pueden cambiar epochs.
- **F:** El grupo llega a 5.52 h antes del margen; excluir también tareas por residuo.
- **I:** Histórico suma no_attention + attention linear + attention mlp; máximo 1.17 h.

## Límites de memoria y comparabilidad

La configuración actual usa jobacct_gather/cgroup, TaskPlugin=task/cgroup y JobAcctGatherParams=(null). Muchos MaxRSS de tareas grandes coinciden casi exactamente con el límite solicitado. No interpretar esa coincidencia como una medición del mínimo de memoria anónima necesario. En cgroup v2 Slurm incluye caché de archivos salvo no_file_cache; no se verificó la versión del cgroup de cada nodo histórico. Documentación oficial: https://slurm.schedmd.com/cgroup_v2.html y https://slurm.schedmd.com/slurm.conf.html.

Además, EC no_attention histórico leía embeddings por residuo y calculaba el pooled durante la carga. El ejemplo orig42_ec_na_4742654_0.out empleó 1007.6 s solo en cargar/poolizar. El plan v3 utiliza embeddings_models_output_mean_pooled.json; es una razón concreta para esperar menos I/O y posiblemente menos RAM, pero no cuantifica por sí sola el nuevo máximo. Los resultados del piloto pooled pequeño tampoco miden EC.

Para residue_store_mode=auto, el código estima sum(longitud_secuencia) × dimensión × 4 / 2^30 GiB y materializa si cabe dentro de 0.8 × presupuesto disponible. Reducir --mem puede cambiar la ruta a lazy y aumentar I/O/duración. Es necesario mantener registrada la ruta real. En el piloto active_sites/ankh_base/no_attention/linear, el job reservado a 200G registró 79.90 GiB y 25m11s; no representa las familias MLP/XGB ni embeddings de dimensión mayor.

Las reservas orientativas de RAM no son mínimos certificados. Evitar recortar RAM en todos los jobs grandes basándose únicamente en MaxRSS. El recorte de tareas pequeñas es mejor respaldado por este historial; EC con mean-pooled necesita su propia medición operativa en los primeros jobs definitivos.

## Operación propuesta sin otro piloto

Congelar el código/configuración y lanzar la corrida definitiva dividida por tarea/familia/split. Agrupar índices de embedding de necesidades similares; un array comparte --mem y --time. No usar una única reserva máxima para todos los embeddings. Mantener 4 CPUs como referencia histórica, y medir antes de atribuir aceleración a más CPUs. Evitar juntar G3 con un único MODEL_PREFIX: binding debe seleccionar linear, mientras active_sites necesita las tres familias; requiere dos envíos o una orquestación explícita. F debe excluir EC y ambas tareas por residuo.

Aplicar bandas 3/12/24/48 h según las particiones disponibles; una petición de 6 h corresponde a b2, no b1. Capturar elapsed, MaxRSS, estado, modo de carga y avance de épocas de los primeros jobs definitivos, y adaptar solo reservas de jobs aún pendientes cuando el cambio no afecte la ruta de datos. Conservar la identidad del protocolo al reanudar: no cambiar hiperparámetros ni código entre tramos.

No se enviaron entrenamientos ni se modificó el plan de producción en esta revisión.
