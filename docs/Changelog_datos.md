# Registro de cambios del dataset

**Proyecto:** 13MBID-ABR2526 – Actividad Práctica I  
**Repositorio:** https://github.com/CristhyanEnciso/13MBID-ABR2526  
**Control de versiones:** DVC v3.59.1  
**Ubicación de datos:** `/data/`  
**Fecha de creación:** 2025-11-08  

---

## Versión inicial del dataset

**Fecha:** 2025-11-08  
**Archivos:**  
- `data/raw/bank-additional-full.csv`  
- `data/raw/bank-additional-new.csv`  

**Acciones realizadas:**
- Descarga desde UCI Machine Learning Repository.  
- Validación de integridad de archivos.  
- Incorporación al repositorio bajo control DVC (`dvc add`).  
- Configuración de almacenamiento remoto `.datastorage`.  
- Creación del archivo de descripción `docs/Descripcion_dataset.md`.

**Responsable:** Cristhyan Enciso  
**Estado:** Dataset original validado y controlado por versión.

---

## Transformaciones previstas (futuras etapas)

| Fase | Descripción | Responsable | Estado | Fecha estimada |
|------|--------------|-------------|---------|----------------|
| Limpieza de los datos | Eliminación de duplicados y valores nulos. | Cristhyan Enciso | Pendiente | 2025-11-09 |
| Integración de datos | Unión de datasets y generación de variables derivadas. | Cristhyan Enciso | Pendiente | 2025-11-10 |
| Formateo de datos | Normalización y codificación de atributos. | Cristhyan Enciso | Pendiente | 2025-11-11 |

---

### Observaciones generales
Este registro será actualizado tras cada iteración de procesamiento, reflejando:
- Archivos modificados.
- Scripts utilizados (`src/`).
- Hash o versión DVC correspondiente.
- Resultados exportados a `/data/processed/` o `/data/interim/`.

---

> **Nota:** Este documento forma parte del sistema de trazabilidad del pipeline de datos y complementa el control automatizado mediante DVC y Git.

---

### US05 – Recolección de datos iniciales

- Se integraron los datasets en la carpeta `data/raw/`:
  - `bank-additional-full.csv`
  - `bank-additional-new.csv`
- Los archivos fueron añadidos bajo control DVC y verificados con `dvc status -c`.
- Se configuró el almacenamiento remoto local `.datastorage` para garantizar la reproducibilidad.
- Se registraron los hashes generados por DVC:
  - `f6cb2c1256ffe2836b36df321f46e92c`
  - `82dfffaa263dd47a08b969ffece9a8d9`
- Se creó la estructura base de carpetas (`data/`, `src/`, `notebooks/`, `docs/`, `reports/`).
- Se añadió un registro inicial de cambios (`docs/Changelog_datos.md`).

**Commit asociado:** `a3b8d5f`  
**Tag:** `v0.5`

---

### US06 – Descripción de los datos (Análisis descriptivo automatizado)

- Se generaron resúmenes descriptivos del dataset `bank-additional-full.csv`:
  - `summary_describe.csv`
  - `summary_dtypes.csv`
  - `summary_nulls.csv`
  - `summary_cardinality.csv`
  - `summary_value_counts.csv`
- Se añadieron figuras de distribución e histogramas en `reports/figures/eda/`.
- Se creó nota metodológica sobre la variable `duration`.
- Scripts involucrados:
  - `src/describe_data.py`
- Control de versión mediante DVC (`reports/summary/`).

**Commit asociado:** `f3665a2`  
**Tag:** `v0.6`

---
