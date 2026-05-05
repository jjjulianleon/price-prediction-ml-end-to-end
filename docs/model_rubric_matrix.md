# Model Rubric Matrix

| Modelo / familia | Exigido por rubrica | Estado en repo | Tipo de entrenamiento | Dependencia | Evidencia principal | Observaciones |
| :-- | :--: | :-- | :-- | :-- | :-- | :-- |
| DummyRegressor | no | implementado | sample | sklearn | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | baseline minimo |
| RidgeRegressor | no | implementado | sample | sklearn | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | baseline lineal regularizado |
| SGDRegressor | sugerido | implementado | incremental | sklearn | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | ruta out-of-core real |
| RandomForestRegressor | sugerido | implementado | sample | sklearn | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | viable solo sobre muestra |
| AdaBoostRegressor | si | implementado | sample | sklearn | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | cobertura de boosting requerida |
| GradientBoostingRegressor | si | implementado | sample | sklearn | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | cobertura de boosting requerida |
| HistGradientBoostingRegressor | no | implementado | sample | sklearn | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | boosting hist adicional |
| XGBoost | si | implementado | sample | `xgboost` | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | requerido cuando la dependencia esta disponible |
| LightGBM | si | implementado | sample | `lightgbm` | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | requerido cuando la dependencia esta disponible |
| CatBoost | si | implementado | sample | `catboost` | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | requerido cuando la dependencia esta disponible |
| Bagging | no | implementado | sample | sklearn | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | cobertura de ensambles |
| Pasting | no | implementado | sample | sklearn | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | cobertura de ensambles |
| Voting | no | implementado | sample | sklearn | `src/models/train_model.py`, `notebooks/04_model_experimentation.ipynb` | ensamble heterogeneo |

## Interpretacion

- `implementado` significa que el modelo existe en el catalogo y entra a la comparacion cuando sus dependencias estan disponibles.
- `incremental` significa compatible con entrenamiento por lotes real en la ruta actual.
- `sample` significa que el modelo se entrena sobre una muestra controlada, manteniendo evaluacion por lotes sobre validation y test.
