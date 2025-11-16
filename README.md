# 🖼️ Clasificación de Imágenes CIFAR-10 con CNNs

> Práctica 3: Desarrollo sistemático de redes neuronales convolucionales para clasificación de imágenes

---

## 📊 Resultados Principales

| Métrica | Valor |
|---------|-------|
| **Mejor Modelo** | CNN Augmentation (2 bloques) |
| **Test Accuracy** | **76-78%** |
| **Parámetros** | 122,000 |
| **Tiempo/Época** | ~4.2s |
| **Mejora vs Baseline** | +26 puntos porcentuales |

### Evolución de Modelos

```
MLP Baseline        →  50% accuracy  (789k parámetros)
CNN Simple (2B)     →  72% accuracy  (122k parámetros) ✨ +22%
CNN + L2            →  74% accuracy  (+regularización)
CNN + Augmentation  →  78% accuracy  (+4% más crítico) 🏆
CNN Deep (3B)       →  78% accuracy  (rendimientos decrecientes)
```

---

## 🎯 Características del Proyecto

### ✨ Técnicas Implementadas

- ✅ **Redes Neuronales Convolucionales** (2 y 3 bloques)
- ✅ **Data Augmentation** (rotación, zoom, traslación, flip)
- ✅ **Regularización L2** (λ=1e-4)
- ✅ **Dropout** (rate=0.5)
- ✅ **Learning Rate Scheduling** (ReduceLROnPlateau, CosineDecay)
- ✅ **Early Stopping** con restauración de mejores pesos
- ✅ **Comparación de Optimizadores** (Adam vs SGD+Momentum)
- ✅ **Estudio de Ablación** (cuantificación de contribuciones)
- ✅ **Análisis de Errores** (matriz de confusión, visualización)

### 📈 Experimentación Sistemática

| Experimento | Objetivo | Resultado Clave |
|-------------|----------|----------------|
| **PROMPT 1** | Setup y carga de datos | 40k train / 10k valid / 10k test |
| **PROMPT 2** | MLP Baseline | 50% accuracy → necesidad de CNNs |
| **PROMPT 3** | CNN Simple | 72% accuracy (+22% vs MLP) |
| **PROMPT 4** | +L2 + Early Stop | 74% accuracy (reduce overfitting) |
| **PROMPT 5** | +Augmentation + ReduceLR | 78% accuracy ⭐ |
| **PROMPT 6** | CNN Profunda (3B) | 78% (rendimientos decrecientes) |
| **PROMPT 7** | Análisis de errores | Gato↔Perro confusión principal |
| **PROMPT 8** | SGD vs Adam | SGD ligeramente mejor (+0.3%) |
| **PROMPT 9** | Estudio de ablación | Augmentation es la técnica más crítica (-4.2%) |
| **PROMPT 10** | Informe y release | Documentación completa |

---

## 🏆 Release Oficial

### 📦 [v1.0-P3-CIFAR10](https://github.com/dwp28/IA_P3_CIFAR10_WillsonDaniel/releases/tag/v1.0)

**Contenido del release:**
- 📓 Notebook completo ejecutable (`.ipynb` + `.pdf`)
- 📊 Resultados completos (`params.yaml`, `metrics.json`, `history_*.csv`)
- 📸 Figuras de análisis (curvas, matriz de confusión, errores)
- 📄 Informe ejecutivo (2 páginas)
- 🔧 Archivos de entorno (`requirements.txt`, `ENVIRONMENT.md`)
- 📦 **`entrega.zip`**: Paquete completo listo para entrega

---

## 📁 Estructura del Proyecto

```
IA_P3_CIFAR10/
│
├── 📓 notebooks/
│   └── CIFAR10_CNN_Willson.ipynb    # Notebook principal con todos los prompts
│
├── 📊 results/
│   ├── params.yaml                      # Configuraciones de todos los experimentos
│   ├── metrics.json                     # Métricas completas por modelo
│   ├── data_meta.json                   # Metadata y hash de datos
│   ├── history_mlp.csv                  # Historial MLP baseline
│   ├── history_cnn.csv                  # Historial CNN simple
│   ├── history_cnn_l2.csv               # Historial CNN + L2
│   ├── history_aug_reduceLR.csv         # Historial CNN + Augmentation
│   ├── history_cnn_deep_3blocks.csv     # Historial CNN profunda
│   ├── history_sgd_cosine.csv           # Historial SGD + CosineDecay
│   ├── classification_report.csv        # Métricas por clase
│   ├── tabla_ablacion.csv               # Resultados del estudio de ablación
│   └── comparacion_*.csv                # Tablas comparativas
│
├── 📸 figuras/
│   ├── *_mlp_curvas.png                 # Curvas de aprendizaje MLP
│   ├── *_cnn_curvas.png                 # Curvas CNN simple
│   ├── *_cnn_l2_early_curvas.png        # Curvas CNN + L2
│   ├── *_aug_reduceLR_curvas.png        # Curvas CNN + Augmentation
│   ├── *_cnn_deep_3blocks_curvas.png    # Curvas CNN profunda
│   ├── *_confusion_matrix_*.png         # Matriz de confusión
│   ├── *_errores_tipicos_*.png          # Visualización de errores
│   ├── *_adam_vs_sgd_comparison.png     # Comparación optimizadores
│   ├── *_ablation_study.png             # Estudio de ablación
│   └── visualizacion_cifar10.png        # Ejemplos del dataset
│
├── 💾 outputs/
│   ├── entrega.zip                      # Paquete completo para entrega
│   ├── CIFAR10_CNN_Willson.ipynb     # Notebook descargado
│   ├── CIFAR10_CNN_Willson.pdf       # PDF del notebook
│   └── best_model_weights.h5            # Pesos del mejor modelo
│
├── 🔧 env/
│   ├── requirements.txt                 # Dependencias congeladas
│   └── ENVIRONMENT.md                   # Versiones del sistema
│
├── 📄 README.md                         # Este archivo
├── 📄 .gitignore                        # Archivos ignorados por Git
└── 📄 LICENSE                           # Licencia del proyecto
```

---

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.10+
- TensorFlow 2.15+
- GPU (opcional, recomendado para entrenamiento)

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/dwp28/IA_P3_CIFAR10_WillsonDaniel.git
cd IA_P3_CIFAR10_WillsonDaniel

# 2. Checkout del release estable
git checkout v1.0-P3-CIFAR10

# 3. Instalar dependencias
pip install -r env/requirements.txt

# 4. Abrir notebook en Colab o Jupyter
jupyter notebook notebooks/CIFAR10_CNN_WillsonDaniel.ipynb
```

### Ejecución en Google Colab

1. Ve a [Google Colab](https://colab.research.google.com/)
2. Archivo → Abrir cuaderno → GitHub
3. Pega la URL de este repositorio
4. Selecciona el notebook `CIFAR10_CNN_WillsonDaniel.ipynb`
5. Runtime → Run all

---

## 🔬 Reproducibilidad

### Configuración Garantizada

```python
# Semilla fijada en todas las librerías
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

### Entorno Documentado

- **Python:** 3.10.12
- **TensorFlow:** 2.15.0
- **GPU:** Tesla T4 (15 GB VRAM)
- **Sistema:** Google Colab (Ubuntu 22.04)

### Verificación de Integridad

```bash
# Hash SHA-256 de datos (primeras 1024 imágenes train)
# Documentado en: results/data_meta.json
```

### Ejecución Completa

```bash
# El notebook tarda aproximadamente 45-60 minutos en ejecutarse completo
# (incluye entrenamiento de 5+ modelos)

# Runtime → Factory reset runtime (limpiar memoria)
# Runtime → Run all (ejecutar todas las celdas)
```

---

## 📊 Resultados Detallados

### Matriz de Confusión (Mejor Modelo)

**Confusiones principales:**

| Par de Clases | Errores Mutuos | Razón |
|---------------|----------------|-------|
| 🐱 Gato ↔ 🐕 Perro | 85 | Cuadrúpedos similares, detalles perdidos a 32×32px |
| 🚗 Auto ↔ 🚚 Camión | 62 | Formas rectangulares, difícil distinguir tamaño |
| 🦌 Ciervo ↔ 🐴 Caballo | 48 | Proporciones similares, colores terrosos |

**Clases más fáciles:**
- 🚢 Barco (F1=0.87) - Fondo de agua distintivo
- ✈️ Avión (F1=0.86) - Forma única con alas
- 🐸 Rana (F1=0.85) - Color verde único

### Estudio de Ablación

**Ranking de importancia de técnicas:**

1. 🥇 **Data Augmentation** → -4.2% sin ella (la más crítica)
2. 🥈 **L2 Regularization** → -1.7% sin ella
3. 🥉 **Dropout** → -1.3% sin ella

**Conclusión:** Data Augmentation multiplica el dataset efectivamente 5-10×, siendo indispensable con solo 40k imágenes de entrenamiento.

### Comparación de Optimizadores

| Optimizador | Convergencia | Estabilidad | Test Acc |
|-------------|--------------|-------------|----------|
| Adam + ReduceLR | Rápida (3-5 épocas) | Oscilaciones | 78.2% |
| SGD + CosineDecay | Lenta (7-10 épocas) | Muy suave | 78.5% |

**Recomendación:** Adam para prototipado rápido; SGD para modelos finales en producción.

---

## 📖 Documentación Completa

### Informe Ejecutivo

El informe completo de 2 páginas está incluido en el notebook principal e incluye:

- ✅ Problema y datos (CIFAR-10, splits, preprocesamiento)
- ✅ Metodología (evolución de arquitecturas)
- ✅ Resultados principales (tablas, figuras, métricas)
- ✅ Cinco decisiones justificadas técnicamente
- ✅ Limitaciones y próximos pasos
- ✅ Recuadro de reproducibilidad completo

### Archivos Clave de Trazabilidad

| Archivo | Contenido |
|---------|-----------|
| `results/params.yaml` | Configuraciones de todos los experimentos |
| `results/metrics.json` | Métricas completas (val_acc, test_acc, best_model) |
| `results/data_meta.json` | Hash de datos, formas, normalización |
| `results/history_*.csv` | Historial época por época de cada modelo |
| `env/requirements.txt` | Dependencias congeladas |
| `env/ENVIRONMENT.md` | Versiones de Python, TensorFlow, GPU |

---

## 🎓 Hallazgos Clave

### 1. CNNs superan dramáticamente a MLPs en visión (+22%)

**Razón:** Sesgo inductivo espacial + compartición de pesos + invariancia translacional

### 2. Data Augmentation es la técnica más impactante (+4%)

**Razón:** Multiplica dataset 5-10× efectivamente, indispensable con 40k imágenes

### 3. Más profundidad ≠ Siempre mejor (rendimientos decrecientes)

**Razón:** CNN 3B tiene +100% parámetros y +50% tiempo pero solo +0-1% accuracy

### 4. Gato/Perro son las clases más difíciles de distinguir

**Razón:** Cuadrúpedos con proporciones similares; detalles faciales perdidos a 32×32px

### 5. Adam converge más rápido, SGD encuentra mejores mínimos

**Razón:** Adam adapta LR por parámetro (rápido); SGD+momentum explora mejor (robusto)

---

## 🔮 Mejoras Futuras Propuestas

### 1️⃣ Label Smoothing + Focal Loss

- **Mejora esperada:** +1.5-2.5% → 79-81% accuracy
- **Esfuerzo:** Bajo (20-30 líneas de código)
- **Beneficio:** Reduce overconfidence, focaliza en clases difíciles

### 2️⃣ Transfer Learning (MobileNetV2)

- **Mejora esperada:** +6-9% → 83-87% accuracy
- **Esfuerzo:** Moderado (40-60 líneas)
- **Beneficio:** Aprovecha ImageNet (1.2M imágenes pre-entrenadas)

### 3️⃣ Otras técnicas avanzadas

- Mixup / CutMix para mayor robustez
- AutoAugment para políticas óptimas automáticas
- Ensemble de 3-5 modelos
- Arquitecturas modernas (EfficientNet, Vision Transformer)

---

## 📝 Citación

Si utilizas este proyecto o código, por favor cita:

```bibtex
@misc{cifar10_cnn_2025,
  author = {Daniel WP},
  title = {Clasificación de Imágenes CIFAR-10 con CNNs},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dwp/IA_P3_CIFAR10_WillsonDaniel}
}
```

---

## 👤 Autor

**Daniel Willson Pastor**

- GitHub: dwp28(https://github.com/dwp28)
- Email: 
- Universidad: UNIE

---

## 📅 Timeline del Proyecto

- **Inicio:** Noviembre 2025
- **PROMPT 1-3:** Setup + Baselines (MLP, CNN simple)
- **PROMPT 4-6:** Regularización + Arquitecturas profundas
- **PROMPT 7-9:** Análisis + Ablación + Optimizadores
- **PROMPT 10:** Informe final + Release
- **Release v1.0:** Noviembre 2025

---

## 📜 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

## 🙏 Agradecimientos

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) por Alex Krizhevsky, Vinod Nair, Geoffrey Hinton
- **Plataforma:** [Google Colab](https://colab.research.google.com/) por recursos GPU gratuitos
- **Frameworks:** [TensorFlow](https://www.tensorflow.org/) y [Keras](https://keras.io/)
- **Comunidad:** Stack Overflow, TensorFlow Docs, Papers with Code

---

## 📞 Contacto y Soporte

¿Preguntas, bugs o sugerencias?

- 🐛 **Issues:** [GitHub Issues](https://github.com/dwp/IA_P3_CIFAR10_WillsonDaniel/issues)
- 💬 **Discusiones:** [GitHub Discussions](https://github.com/dwp28/IA_P3_CIFAR10_WillsonDaniel/discussions)
- 📧 **Email:**

---

<div align="center">

**⭐ Si este proyecto te ha sido útil, considera darle una estrella en GitHub ⭐**


</div>

---

<div align="center">
  <sub>Desarrollado con ❤️ usando TensorFlow y Google Colab</sub>
</div>