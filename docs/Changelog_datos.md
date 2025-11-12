# Registro de cambios del dataset

**Proyecto:** 13MBID-ABR2526 – Actividad Práctica I  
**Repositorio:** https://github.com/CristhyanEnciso/13MBID-ABR2526  
**Control de versiones:** DVC v3.59.1  
**Ubicación de datos:** `/data/`  
**Fecha de creación:** 2025-11-08  
**Estado actual:** Iteración 1 (Comprensión + Preparación) **CERRADA** – 2025-11-11  

---

## Tabla cronológica de versiones (confirmada desde Git)

| Fecha (-0500)              | Tag                         | Commit   | Descripción breve                                                             |
|----------------------------|-----------------------------|----------|-------------------------------------------------------------------------------|
| 2025-11-08 11:22:16        | **data-v1.0-inicial**       | 767d742  | Ingesta y versionado inicial de datos crudos (DVC add).                       |
| 2025-11-08 21:41:29        | **v0.6**                    | dd070c1  | Descripción automatizada del dataset (`describe_data.py`).                    |
| 2025-11-08 22:04:48        | **v0.7**                    | f281869  | EDA visual reproducible (`visualize_data`).                                   |
| 2025-11-09 00:32:37        | **v0.8**                    | 003531e  | Verificación de calidad (Pandera + Pytest).                                   |
| 2025-11-09 11:08:51        | **v0.9**                    | 0e835cf  | Selección de atributos relevantes.                                            |
| 2025-11-09 17:17:53        | **v0.10**                   | 39b26be  | Limpieza de datos (duplicados / nulos) + `data_loss_report.csv`.              |
| 2025-11-09 20:54:16        | **v0.11**                   | 9736b06  | Construcción de datos (`build_features`) – `banking_features.csv`.            |
| 2025-11-09 21:38:44        | **v0.12**                   | 624c3e6  | Integración de datos – `bank_final.csv` + `integration_summary.csv`.          |
| 2025-11-09 23:17:08        | **v0.13**                   | 130e9ea  | Formateo final – `bank_formatted.csv` + reportes de formato.                  |

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
**Fecha:** 2025-11-08 | **Commit:** `dd070c1`  
**Script:** `src/describe_data.py`  
**Salidas:** `reports/summary/*.csv`, `reports/figures/desc/*.png`  
**Repro:** `dvc repro describe_data`  

---

## v0.7 – Exploración de datos (EDA)  
**Fecha:** 2025-11-08 | **Commit:** `f281869`  
**Script:** `src/data_visualization.py`  
**Salidas:** `reports/figures/eda/`, `reports/summary/`  

---

## v0.8 – Verificación de calidad de datos  
**Fecha:** 2025-11-09 | **Commit:** `003531e`  
**Validaciones:** `tests/test_data_quality.py`, `tests/test_data_gx.py`  
**Outputs:** `docs/test_results/*`  
**Repro:** `dvc repro test_data_quality` y `dvc repro test_data_gx`  

---

## v0.9 – Selección de atributos  
**Fecha:** 2025-11-09 | **Commit:** `0e835cf`  
**Script:** `src/select_features.py`  
**Salidas:** `data/interim/banking_selected.csv`, `reports/selection/*.csv`  
**Repro:** `dvc repro select_features`  

---

## v0.10 – Limpieza de datos  
**Fecha:** 2025-11-09 | **Commit:** `39b26be`  
**Script:** `src/clean_data.py`  
**Salidas:** `data/interim/banking_clean.csv`, `reports/cleaning/data_loss_report.csv`  
**Repro:** `dvc repro clean_data`  

---

## v0.11 – Construcción de datos  
**Fecha:** 2025-11-09 | **Commit:** `9736b06`  
**Script:** `src/build_features.py`  
**Salidas:** `data/interim/banking_features.csv`, `reports/features/*`  
**Repro:** `dvc repro build_features`  

---

## v0.12 – Integración de datos  
**Fecha:** 2025-11-09 | **Commit:** `624c3e6`  
**Script:** `src/integrate_data.py`  
**Salidas:** `data/processed/bank_final.csv`, `reports/integration/integration_summary.csv`  
**Repro:** `dvc repro integrate_data`  

---

## v0.13 – Formateo final  
**Fecha:** 2025-11-09 | **Commit:** `130e9ea`  
**Script:** `src/format_data.py`  
**Salidas:** `data/processed/bank_formatted.csv`, `reports/format/format_summary.txt`, `reports/format/formatted_columns.csv`  
**Repro:** `dvc repro format_data`  

---

## Cierre de Iteración 1 – v1.0_iteracion1_done  
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

> Este changelog consolida la trazabilidad del flujo **CRISP-DM** bajo el enfoque **Scrum + DataOps + MLOps**,  
> garantizando **reproducibilidad, control de versión y alineación metodológica** para las fases posteriores.
