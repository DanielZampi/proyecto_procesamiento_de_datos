# 🔮 ProphetNet — Generación Automática de Titulares de Noticias

> Proyecto Final — Procesamiento de Datos Secuenciales  
> Arquitectura Transformer encoder-decoder con *future n-gram prediction*

---
## Integrantes

* Jairo Andrés Pérez Hurtatis
* Diego Mauricio Ortiz
* Daniel Felipe Zamora Pineda

---

## 1. Resumen (Abstract)

Este proyecto implementa el proceso de inferencia sobre **ProphetNet**, un modelo Transformer encoder-decoder desarrollado por Microsoft, aplicado a la tarea de generación automática de titulares a partir de artículos de noticias en inglés (*headline generation*). Se utilizan los pesos preentrenados del modelo `microsoft/prophetnet-large-uncased-cnndm`, ajustado sobre el dataset CNN/DailyMail. La implementación incluye el pipeline completo de inferencia (tokenización → generación con beam search → decodificación), extracción y visualización del mecanismo de cross-attention (Q, K, V), evaluación cuantitativa con métricas ROUGE, y una interfaz interactiva desarrollada con Streamlit que permite ingresar cualquier artículo y observar la inferencia del modelo en tiempo real. Los resultados obtenidos muestran un ROUGE-1 promedio de 0.331, ROUGE-2 de 0.080 y ROUGE-L de 0.288 sobre cinco artículos de referencia.

---

## 2. Introducción

### Artículo base
**ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training**  
Yan et al. (2020) — arXiv:2001.04063  
Repositorio original: https://github.com/microsoft/ProphetNet  
Modelo en HuggingFace: https://huggingface.co/microsoft/prophetnet-large-uncased-cnndm

### Contexto y motivación
La generación automática de titulares es una tarea de procesamiento de datos secuenciales que consiste en producir un resumen conciso de un texto largo. Los modelos Transformer encoder-decoder son el estado del arte para este tipo de tareas, pero presentan un problema fundamental: durante el entrenamiento, el decoder aprende a predecir solo el siguiente token, lo que lleva al modelo a enfocarse demasiado en el contexto local inmediato (*overfitting on local context*) y perder coherencia global.

### Objetivo
Aplicar la arquitectura ProphetNet para inferencia sobre artículos de noticias en inglés, comprender en profundidad su mecanismo de atención n-stream, y presentar los resultados mediante una interfaz interactiva que visualice el proceso de generación.

---

## 3. Marco Teórico

### 3.1 Arquitectura Transformer encoder-decoder

ProphetNet sigue la arquitectura estándar encoder-decoder:

- **Encoder**: procesa el artículo de entrada de forma bidireccional, similar a BERT. Cada token atiende a todos los demás tokens del artículo, construyendo representaciones ricas en contexto. Al final produce un conjunto de *hidden states* que capturan el significado del texto completo.

- **Decoder**: genera el titular token por token de forma autoregresiva, usando dos tipos de atención:
  - **Self-attention**: el decoder atiende a los tokens que ya generó.
  - **Cross-attention**: el decoder consulta los hidden states del encoder para incorporar información del artículo original.

### 3.2 Mecanismo de atención — Q, K, V

El mecanismo de atención opera con tres componentes para cada token:

- **Query (Q)**: vector que representa "qué información está buscando" el token actual.
- **Key (K)**: vector que representa "de qué trata" la información de cada token.
- **Value (V)**: vector que contiene el contenido real de la información de cada token.

Cada uno se obtiene multiplicando el hidden state por matrices de pesos aprendidas durante el entrenamiento:

```
Q = h · Wq
K = h · Wk  
V = h · Wv
```

La fórmula de atención es:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
```

El producto `Q·Kᵀ` mide la similitud entre el query de un token y los keys de todos los demás. El `softmax` normaliza esos scores en probabilidades (los *attention weights*), que luego ponderan los Values para producir la representación final.

### 3.3 Innovación de ProphetNet: N-stream decoder

La innovación clave de ProphetNet es su **n-stream self-attention** en el decoder. En vez de tener un único flujo que predice solo el siguiente token, ProphetNet usa **n flujos en paralelo**, donde cada stream predice un token futuro diferente.

Con n=2 (el valor usado en este proyecto), para cada posición `t` el decoder mantiene:
- **Stream 1**: predice el token `t+1`, con matrices propias `Wq1, Wk1, Wv1`
- **Stream 2**: predice el token `t+2`, con matrices propias `Wq2, Wk2, Wv2`

Esto cambia la función de pérdida durante el entrenamiento:

**Transformer estándar:**
```
L = −Σ log P(yₜ | y<t, x)
```

**ProphetNet (n=2):**
```
L = −Σ Σᵢ log P(yₜ₊ᵢ | y<t, x)
```

Al obligar al modelo a "planear" varios tokens hacia adelante durante el entrenamiento, aprende representaciones más ricas y genera texto más coherente.

### 3.4 Parámetros del modelo

| Parámetro | Valor |
|---|---|
| Parámetros totales | 485,085,184 |
| Capas del encoder | 12 |
| Capas del decoder | 12 |
| Dimensión oculta (d_model) | 1024 |
| Cabezas de atención | 16 |
| N-gram del decoder | 2 |
| Vocabulario | 30,522 tokens |

---

## 4. Metodología

### 4.1 Modelo utilizado
Se utilizó `microsoft/prophetnet-large-uncased-cnndm`, la versión fine-tuneada sobre el dataset CNN/DailyMail específicamente para headline generation. Esta versión produce resultados significativamente mejores que el modelo base (`prophetnet-large-uncased`) para esta tarea.

### 4.2 Herramientas
- **Python 3.12** — lenguaje de implementación
- **PyTorch** — framework de deep learning
- **HuggingFace Transformers** — carga del modelo y tokenizer
- **rouge-score** — cálculo de métricas de evaluación
- **matplotlib / seaborn** — visualización de resultados
- **Streamlit** — interfaz interactiva
- **Google Colab (GPU T4)** — entorno de ejecución

### 4.3 Pesos preentrenados
Los pesos se descargan automáticamente desde HuggingFace Hub con:

```python
from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer

tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased-cnndm")
model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased-cnndm")
```

No se realiza entrenamiento desde cero — se usan directamente los pesos publicados por Microsoft.

---

## 5. Desarrollo e Implementación

### 5.1 Clonar el repositorio

```bash
git clone https://github.com/DanielZampi/proyecto_procesamiento_de_datos.git
cd proyecto_procesamiento_de_datos
```

### 5.2 Instalar dependencias

```bash
pip install transformers torch sentencepiece rouge-score matplotlib seaborn streamlit
```

### 5.3 Correr el notebook de inferencia

Abrir `ProphetNet_Completo.ipynb` en Google Colab:
1. Ir a `Entorno de ejecución → Cambiar tipo → GPU (T4)`
2. Ejecutar todas las celdas en orden

### 5.4 Correr la app interactiva

```bash
streamlit run app.py
```

O acceder directamente en: **https://proyectoprocesamientodedatos.streamlit.app**

### 5.5 Pipeline de inferencia

El proceso de inferencia sigue tres pasos:

**Preprocesamiento** — el texto se convierte a minúsculas (modelo uncased) y se tokeniza:
```python
inputs = tokenizer(articulo.lower(), return_tensors="pt", truncation=True, max_length=512)
```

**Inferencia** — el modelo genera el titular con beam search:
```python
output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    num_beams=8,
    max_length=60,
    min_length=8,
    no_repeat_ngram_size=3,
    early_stopping=True
)
```

**Postprocesamiento** — los tokens generados se decodifican a texto:
```python
titular = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### 5.6 Extracción del mecanismo de atención

Se usa `output_attentions=True` para obtener los pesos de cross-attention internos:

```python
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    decoder_input_ids=labels,
    output_attentions=True
)

# Última capa, promedio de las 16 cabezas
cross_attn = outputs.cross_attentions[-1].squeeze(0).mean(dim=0).cpu().numpy()

# Conversión de log-space a probabilidades reales
cross_attn = np.exp(cross_attn)
cross_attn = cross_attn / cross_attn.sum(axis=-1, keepdims=True)
```

---

## 6. Resultados y Análisis

### 6.1 Titulares generados (beam=8, max_length=60)

| Artículo | Titular real | ProphetNet |
|---|---|---|
| Diplomacia Bolivia | us rejects charges against its ambassador in bolivia | the us state department said it had received no formal word from bolivia that it was expelling the us ambassador |
| Dinosaurio Argentina | scientists discover one of the largest dinosaurs ever found in argentina | scientists have discovered a new species of dinosaur in argentina. the titanosaur is estimated to have weighed 70 tons |
| Apple ganancias | apple reports record quarterly revenue driven by iphone and services growth | apple announces record revenue of 90 billion dollars. the company announced a new stock buyback program |
| NASA Marte | nasa perseverance rover collects first mars rock sample | nasa's perseverance rover has successfully collected its first rock sample from the surface of mars |
| OMS emergencia | who declares global health emergency over new respiratory virus | the world health organization declared a global health emergency on thursday. a new respiratory virus continues to spread |

### 6.2 Métricas ROUGE

| Ejemplo | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Diplomacia Bolivia | 0.2703 | 0.0000 | 0.1622 |
| Dinosaurio Argentina | 0.3684 | 0.0556 | 0.3158 |
| Apple ganancias | 0.1935 | 0.0000 | 0.1935 |
| NASA Marte | 0.4444 | 0.1176 | 0.3889 |
| OMS emergencia | 0.3784 | 0.2286 | 0.3784 |
| **Promedio** | **0.3310** | **0.0804** | **0.2878** |

El ROUGE-1 promedio de 0.331 indica un overlap significativo de palabras individuales entre el titular generado y el real. El ROUGE-2 de 0.080 refleja que las frases exactas difieren — el modelo parafrasea en vez de copiar. El ejemplo de NASA (ROUGE-1: 0.444) es el mejor resultado, donde el modelo capturó con precisión los elementos clave: rover, primera muestra, Marte.

### 6.3 Análisis del mecanismo de atención

El heatmap de cross-attention de la última capa del decoder muestra que los tokens más informativos del artículo reciben consistentemente los pesos más altos. Para el artículo sobre Irán, los tokens `"executed"`, `"protests"` y `"january"` concentraron la mayor atención, lo que confirma que el modelo identifica correctamente las palabras clave antes de generar cada token del titular.

### 6.4 Análisis de hiperparámetros

Se evaluó el efecto de beam search y max_length sobre el artículo de Irán:

| Beam | Max tokens | Resultado |
|---|---|---|
| 4 | 50 | Completo pero demasiado largo |
| 8 | 30 | Mejor calidad pero truncado |
| **8** | **60** | **Completo, coherente y con punto final ✅** |

La configuración óptima encontrada fue **beam=8, max_length=60**.

---

## 7. Conclusiones

- ProphetNet demuestra ser efectivo para headline generation cuando se usa la versión fine-tuneada sobre CNN/DailyMail, con un ROUGE-1 promedio de 0.331.
- La innovación del n-stream decoder (n=2) obliga al modelo a planear dos tokens hacia adelante durante el entrenamiento, produciendo titulares más coherentes que un decoder estándar.
- El mecanismo de cross-attention extrae correctamente los tokens más relevantes del artículo, lo que se visualiza claramente en los heatmaps generados.
- El modelo presenta limitaciones con textos fuera del dominio periodístico — al pasarle un texto educativo descriptivo, genera un resumen en vez de un titular conciso, lo que refleja el sesgo del dataset de entrenamiento.
- La configuración de hiperparámetros afecta significativamente la calidad del output: beam=8 con max_length=60 produce los mejores resultados para este modelo.

**Posibles mejoras:** fine-tuning sobre datasets en español, exploración de ProphetNet con n=3, y evaluación sobre datasets de noticias más recientes como XSum.

---

## 8. Referencias

[1] Y. Yan, W. Qi, Y. Gong, D. Liu, N. Duan, J. Chen, R. Zhang, and M. Zhou, "ProphetNet: Predicting future n-gram for sequence-to-sequence pre-training," *arXiv preprint arXiv:2001.04063*, 2020.

[2] Microsoft, "microsoft/prophetnet-large-uncased-cnndm," Hugging Face, 2020. [Online]. Available: https://huggingface.co/microsoft/prophetnet-large-uncased-cnndm

[3] C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in *Proc. ACL Workshop on Text Summarization Branches Out*, Barcelona, Spain, 2004, pp. 74–81.

[4] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[5] K. M. Hermann et al., "Teaching machines to read and comprehend," in *Advances in Neural Information Processing Systems*, vol. 28, 2015.
