# Model Rubric Matrix

| Modelo / familia | Exigido por rubrica | Estado en repo | Tipo de entrenamiento | Dependencia | Evidencia principal | Observaciones |
| :-- | :--: | :-- | :-- | :-- | :-- | :-- |
| DummyRegressor | no | shortlist | sample | sklearn | `src/models/experiment_runner.py`, `notebooks/04_model_experimentation.ipynb` | baseline minimo |
| RidgeRegressor | no | archivado | sample | sklearn | `src/models/model_zoo.py`, `notebooks/temp.txt` | dominado por otros baselines |
| SGDRegressor | sugerido | shortlist | incremental | sklearn | `src/models/experiment_runner.py`, `notebooks/04_model_experimentation.ipynb` | ruta out-of-core real |
| RandomForestRegressor | sugerido | shortlist | sample | sklearn | `src/models/experiment_runner.py`, `notebooks/04_model_experimentation.ipynb` | referencia de ensamble no boosting |
| AdaBoostRegressor | si | archivado | sample | sklearn | `src/models/model_zoo.py`, `notebooks/temp.txt` | peor tradeoff que boosters mas fuertes |
| GradientBoostingRegressor | si | shortlist y produccion | sample | sklearn | `src/models/experiment_runner.py`, `src/models/train_model.py`, `notebooks/temp.txt` | mejor `val_rmse` completado al 2026-05-05 |
| HistGradientBoostingRegressor | no | shortlist | sample | sklearn | `src/models/experiment_runner.py`, `notebooks/04_model_experimentation.ipynb` | booster hist adicional |
| XGBoost | si | shortlist | sample | `xgboost` | `src/models/experiment_runner.py`, `notebooks/04_model_experimentation.ipynb` | requerido cuando la dependencia esta disponible |
| LightGBM | si | shortlist | sample | `lightgbm` | `src/models/experiment_runner.py`, `notebooks/temp.txt` | candidato real si el entorno esta estable |
| CatBoost | si | shortlist | sample | `catboost` | `src/models/experiment_runner.py`, `notebooks/temp.txt` | corrida inconclusa por estabilidad, no por falta de soporte |
| Bagging | no | archivado | sample | sklearn | `src/models/model_zoo.py`, `notebooks/temp.txt` | dominado por boosters y random forest |
| Pasting | no | archivado | sample | sklearn | `src/models/model_zoo.py`, `notebooks/temp.txt` | dominado por boosters y random forest |
| Voting | no | archivado | sample | sklearn | `src/models/model_zoo.py`, `notebooks/temp.txt` | costo alto sin ganar la comparacion |

## Interpretacion

- `shortlist` significa que el modelo sigue en el flujo principal de comparacion.
- `archivado` significa que el modelo se conserva en el zoo solo para trazabilidad o reruns excepcionales.
- `incremental` significa compatible con entrenamiento por lotes real en la ruta actual.
- `sample` significa que el modelo se entrena sobre una muestra controlada, manteniendo evaluacion por lotes sobre validation y test.
