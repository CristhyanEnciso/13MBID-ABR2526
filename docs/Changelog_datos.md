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
| Limpieza de los datos | Eliminación de duplicados y valores nulos. | C. Enciso | Pendiente | 2025-11-09 |
| Integración de datos | Unión de datasets y generación de variables derivadas. | C. Enciso | Pendiente | 2025-11-10 |
| Formateo de datos | Normalización y codificación de atributos. | C. Enciso | Pendiente | 2025-11-11 |

---

### Observaciones generales
Este registro será actualizado tras cada iteración de procesamiento, reflejando:
- Archivos modificados.
- Scripts utilizados (`src/`).
- Hash o versión DVC correspondiente.
- Resultados exportados a `/data/processed/` o `/data/interim/`.

---

> **Nota:** Este documento forma parte del sistema de trazabilidad del pipeline de datos y complementa el control automatizado mediante DVC y Git.
