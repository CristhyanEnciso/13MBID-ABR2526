# Registro de cambios del dataset

**Proyecto:** 13MBID-ABR2526 – Actividad Práctica I  
**Repositorio:** https://github.com/CristhyanEnciso/13MBID-ABR2526  
**Control de versiones:** DVC v3.59.1  
**Ubicación de datos:** `/data/`  
**Fecha de creación:** 2025-11-08  
**Estado actual:** Iteración 1 (Comprensión + Preparación) **CERRADA** – 2025-11-11  

---

## Tabla cronológica de versiones (confirmada desde Git)

| Fecha (-0500) | Tag | Commit | Descripción breve |
|----------------|-----|---------|--------------------|
| 2025-11-08 11:22 | **v0.5 (data-v1.0-inicial)** | `<hash ≈ 767d742>` | Ingesta y versionado inicial de datos crudos (DVC add). |
| 2025-11-08 21:41 | **v0.6** | `<hash ≈ f3665a2>` | Descripción automatizada del dataset (`describe_data.py`). |
| 2025-11-08 22:04 | **v0.7** | — | EDA visual reproducible (`visualize_data`). |
| 2025-11-09 00:32 | **v0.8** | — | Verificación de calidad (Pandera + Pytest). |
| 2025-11-09 11:08 | **v0.9** | — | Selección de atributos relevantes. |
| 2025-11-09 17:17 | **v0.10** | — | Limpieza de datos (duplicados / nulos). |
| 2025-11-09 23:17 | **v0.13** | — | Construcción + Integración + Formateo final de datos. |

> Fechas obtenidas con  
> `git log --tags --simplify-by-decoration --pretty="format:%ai %d"`  
> confirmando orden y temporalidad real de la iteración 1.

---

## v0.5 – Versión inicial del dataset  
**Fecha:** 2025-11-08  | **Commit:** `767d742`  
**Archivos:**  
- `data/raw/bank-additional-full.csv`  
- `data/raw/bank-additional-new.csv`  
**Acciones:** descarga UCI ML Repo → validación → `dvc add` → configuración de remoto `./.datastorage`.  
**Hashes DVC:** `f6cb2c1256ffe2836b36df321f46e92c`, `82dfffaa263dd47a08b969ffece9a8d9`  
**Estado:** dataset crudo validado y versionado.  

---

## v0.6 – Descripción automatizada de datos  
**Fecha:** 2025-11-08  | **Commit:** `f3665a2`  
**Script:** `src/describe_data.py`  
**Salidas:** `reports/summary/*.csv`, `reports/figures/desc/*.png`  
**Repro:** `dvc repro describe_data`  

---

## v0.7 – Exploración de datos (EDA)  
**Fecha:** 2025-11-08  | **Script:** `src/data_visualization.py`  
Generación de gráficos y métricas reproducibles → `reports/figures/eda/`, `reports/summary/`.  

---

## v0.8 – Verificación de calidad de datos  
**Fecha:** 2025-11-09  
**Validaciones:** `tests/test_data_quality.py`, `tests/test_data_gx.py`  
**Outputs:** `docs/test_results/*`  
**Repro:** `dvc repro test_data_quality` y `dvc repro test_data_gx`  

---

## v0.9 – Selección de atributos  
**Fecha:** 2025-11-09  
**Script:** `src/select_features.py`  
**Salidas:** `data/interim/banking_selected.csv`, `reports/selection/*.csv`  
**Repro:** `dvc repro select_features`  

---

## v0.10 – Limpieza de datos  
**Fecha:** 2025-11-09  
**Script:** `src/clean_data.py`  
**Salidas:** `data/interim/banking_clean.csv`, `reports/cleaning/data_loss_report.csv`  
**Repro:** `dvc repro clean_data`  

---

## v0.13 – Construcción · Integración · Formateo final  
**Fecha:** 2025-11-09  
**Scripts:** `build_features.py`, `integrate_data.py`, `format_data.py`  
**Salidas:** `data/processed/bank_formatted.csv` + reportes `/reports/features/`, `/reports/format/`.  
**Repro:** `dvc repro build_features` → `dvc repro integrate_data` → `dvc repro format_data`  

---

## 📦 Cierre de Iteración 1 – v1.0_iteracion1_done  
**Fecha:** 2025-11-11  
**Resumen:**  
- Pipeline completo CRISP-DM hasta **Preparación de Datos**.  
- 13 User Stories completadas (Scrum → Definition of Done verificable).  
- Evidencias versionadas con Git y DVC.  
- Entorno preparado para Iteración 2 (Modelado / Evaluación).  

**Comandos clave:**  
```bash
dvc dag               # Visualizar flujo de etapas
dvc repro             # Reproducir pipeline completo
dvc push              # Sincronizar remoto
git tag -l -n1        # Ver tags y descripciones
```

## 🔍 Auditoría y trazabilidad técnica

**Objetivo:** mantener coherencia y reproducibilidad bajo los principios **DataOps/MLOps**.  

### Verificaciones ejecutadas

| Verificación | Herramienta | Resultado |
|---------------|--------------|-----------|
| Integridad de hashes DVC | `dvc status -c` | ✅ En sync |
| Coherencia entre tags Git y etapas DVC | `git log --tags` · `dvc dag` | ✅ Orden verificado |
| Reproducibilidad de pipeline | `dvc repro` | ✅ Sin errores |
| Control de datos remoto | `dvc push/pull` | ✅ Correcto |
| Registro de configuración | `params.yaml` / `dvc.yaml` | ✅ Actualizados |

---

## 🔮 Próximos hitos (Iteración 2)

- Entrenamiento y evaluación de modelos.  
- Registro de experimentos con **MLflow**.  
- Integración **CI/CD** mediante *GitHub Actions*.  
- Monitoreo y validación continua dentro del ciclo **MLOps loop**.  

---

> Este changelog consolida la trazabilidad del flujo **CRISP-DM** bajo el enfoque **Scrum + DataOps + MLOps**,  
> garantizando **reproducibilidad, control de versión y alineación metodológica** para las fases posteriores.
