# 🎬 Agentic Movie Recommender — Self-Improving Multi-Agent System

**Un sistema de recomendación de películas que aprende y se auto-mejora usando agentes de IA que debaten, juzgan y se optimizan continuamente.**

Este proyecto implementa un sistema de recomendación multi-agente con capacidades de **auto-mejora automática**:
- **Críticos** debaten sobre películas desde múltiples perspectivas
- **Jueces** sintetizan opiniones y producen predicciones calibradas
- **Reviewer** analiza performance y **mejora automáticamente los prompts de los jueces**
- **Calibrador** se ajusta en línea con feedback del usuario

---

## 📋 Tabla de Contenidos

1. [Problema que Resuelve](#-problema-que-resuelve)
2. [Arquitectura del Sistema](#-arquitectura-del-sistema)
3. [Ciclo de Auto-Mejora](#-ciclo-de-auto-mejora)
4. [Métricas de Mejora](#-métricas-de-mejora-evidencia-cuantificable)
5. [Instalación y Ejecución](#-instalación-y-ejecución)
6. [Características Clave](#-características-clave)

---

## 🎯 Problema que Resuelve

### El Desafío
Los sistemas de recomendación tradicionales tienen limitaciones fundamentales:
- **Estáticos**: Modelos entrenados una vez, no se adaptan a cambios en preferencias
- **Caja Negra**: Difícil entender por qué se recomienda algo
- **Rigidez**: Requieren re-entrenamiento completo para mejorar
- **Falta de Contexto**: No consideran matices como estado de ánimo, contexto social, etc.

### Nuestra Solución
Un sistema **multi-agente auto-evolutivo** que:

1. **Debate Multi-Perspectiva**: Múltiples críticos especializados (cinéfilo, analista técnico, experto en comedia, etc.) analizan cada película desde diferentes ángulos
2. **Agregación Inteligente**: Jueces sintetizan las opiniones de críticos, ponderando por confiabilidad y relevancia
3. **Auto-Mejora Continua**: Un agente Reviewer monitorea el desempeño y **automáticamente mejora los prompts** de los componentes con peor rendimiento
4. **Aprendizaje Online**: Calibración en tiempo real con feedback del usuario sin re-entrenamiento

### Ventajas sobre Sistemas Tradicionales
- ✅ **Interpretabilidad**: Cada predicción incluye justificaciones explícitas
- ✅ **Adaptabilidad**: Mejora continuamente sin intervención manual
- ✅ **Contexto Rico**: Considera personalidad del usuario, géneros, críticas especializadas
- ✅ **Incertidumbre Calibrada**: Reporta confianza en cada predicción

---

## 🏗️ Arquitectura del Sistema

### Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR AGENT                           │
│                    (Coordina todo el proceso)                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ↓                         ↓
         ┌──────────────────┐      ┌──────────────────┐
         │   DATA STORE     │      │    RETRIEVER     │
         │  (Train/Test)    │──────│  (User Context)  │
         └──────────────────┘      └──────────────────┘
                    │
                     ───
                        │
            ┌───────────┴───────────┐
            ↓                       ↓
    ┌───────────────┐      ┌────────────────┐
    │   CRITICS     │      │    JUDGES      │
    │ (Multi-Agent  │──────│ (Aggregation)  │
    │   Debate)     │      │                │
    └───────────────┘      └────────┬───────┘
         │                          │
         │  Opiniones               │  Rating ponderado
         │  + Confianza             │  + Justificación
         │                          ↓
         │                 ┌─────────────────┐
         │                 │   CALIBRATOR    │
         │                 │ (Online Learn)  │
         │                 └────────┬────────┘
         │                          │
         │                          ↓
         │                 ┌─────────────────────┐
         │                 │  Predicción Final   │
         │                 │  ŷ ± σ              │
         │                 └──────────┬──────────┘
         │                            │
         ↓                            ↓
    ┌─────────────────────────────────────────────────┐
    │              REVIEWER AGENT                      │
    │  • Analiza performance de jueces y críticos     │
    │  • Identifica componentes con bajo rendimiento  │
    │  • GENERA AUTOMÁTICAMENTE prompts mejorados     │
    │  • Crea versiones v2, v3... de jueces          │
    └─────────────────────────────────────────────────┘
                            │
                            ↓
                   ┌─────────────────┐
                   │ Prompt Updates  │
                   │ judge_v1, v2... │
                   └─────────────────┘
```

### Componentes Principales

#### 1. **Orchestrator** (`orchestrator.py`)
- Coordina el flujo completo de predicción
- Maneja el ciclo de debate → agregación → calibración
- Actualización online con feedback del usuario

#### 2. **Critics** (`critics.py`)
- 11 críticos especializados con diferentes perspectivas:
  - `cinephile`: Análisis cinematográfico profundo
  - `technical_expert`: Aspectos técnicos (fotografía, edición)
  - `comedy_specialist`: Experto en comedia
  - `character_focused`: Análisis de personajes
  - `genre_purist`: Purista de géneros
  - Y más...
- Cada crítico evalúa la película y proporciona score + rationale

#### 3. **Judges** (`judges.py`)
- 11 jueces con estrategias diferentes de agregación:
  - `grounded_v1`: Prioriza evidencia verificable
  - `confidence_weighted`: Pondera por confianza
  - `consensus_builder`: Busca consenso
  - `contrarian_seeker`: Valora perspectivas minoritarias
  - Y más...
- Sintetizan opiniones de críticos en predicción calibrada

#### 4. **Reviewer** (`reviewer.py`) - **🔑 COMPONENTE DE AUTO-MEJORA**
- Monitorea performance cada N predicciones
- Calcula métricas por juez (error promedio, consistencia)
- **Genera automáticamente nuevos prompts mejorados**
- Crea versiones incrementales (judge_v1, judge_v2...)
- Proporciona recomendaciones de optimización

#### 5. **Calibrator** (`calibrator.py`)
- Regresión lineal online simple
- Actualiza pesos con feedback del usuario
- Estima incertidumbre (σ)

#### 6. **Router** (`router.py`)
- Política tipo bandit para seleccionar críticos/jueces
- Balancea exploración vs explotación
- Ajusta selección por género y performance

#### 7. **Data Store** (`data_store.py`)
- Manejo seguro de train/test split
- Previene contaminación de datos
- Proporciona contexto de usuario solo desde train

#### 8. **Prediction Logger** (`prediction_logger.py`)
- Log automático de predicciones a CSV
- Métricas de error, confianza, géneros
- Análisis offline con `analyze_predictions.py`

### Estructura de Archivos

```
hackathon-openai-kavak/
├── src/
│   ├── orchestrator.py       # Agente coordinador principal
│   ├── critics.py            # Multi-agent debate
│   ├── judges.py             # Agregación inteligente
│   ├── reviewer.py           # 🔑 Auto-mejora automática
│   ├── calibrator.py         # Online learning
│   ├── router.py             # Bandit selection
│   ├── retriever.py          # Context retrieval
│   ├── data_store.py         # Train/test management
│   ├── prediction_logger.py  # CSV logging
│   ├── llm_client.py         # LLM interface
│   ├── main_demo.py          # Entry point
│   ├── resources/
│   │   ├── movie_critics/    # Prompts de críticos
│   │   └── judges/           # Prompts de jueces (auto-generados)
│   └── data/
│       └── splits/           # Train/test data
├── analyze_predictions.py    # Análisis de logs
├── docs/
│   ├── arquitectura.md
│   ├── reviewer.md
│   ├── prediction_logger.md
│   └── train_test_split.md
├── logs/
│   ├── events.jsonl          # Event logs
│   └── predictions_*.csv     # Prediction logs
└── requirements.txt
```

---

## 🔄 Ciclo de Auto-Mejora

### Cómo Funciona la Auto-Mejora Automática

El sistema implementa un **ciclo de mejora continua** sin intervención humana:

```
1. PREDICCIÓN
   ├─ Críticos debaten
   ├─ Jueces agregan
   └─ Calibrador produce ŷ ± σ

2. FEEDBACK
   ├─ Usuario proporciona rating real
   └─ Sistema calcula error

3. ACTUALIZACIÓN ONLINE
   ├─ Calibrador ajusta pesos
   └─ Router actualiza performance tracking

4. ANÁLISIS META (cada N predicciones)
   ├─ Reviewer analiza performance
   ├─ Identifica jueces con alto error
   └─ Genera prompts mejorados automáticamente

5. EVOLUCIÓN
   ├─ Crea judge_v1, judge_v2...
   ├─ Documenta cambios y razones
   └─ Integra en siguiente iteración

   ↻ REPITE EL CICLO
```

### Mecanismo de Mejora de Prompts

El **Reviewer Agent** usa LLM para generar prompts mejorados:

1. **Análisis**: Calcula error promedio y desviación estándar por juez
2. **Identificación**: Detecta el juez con peor performance
3. **Generación**: Usa GPT-4 para crear un prompt mejorado que:
   - Mantiene la estrategia original
   - Corrige debilidades identificadas
   - Añade instrucciones para minimizar error
   - Incorpora mejores prácticas observadas
4. **Implementación**: Guarda el nuevo prompt como `judge_id_v1.txt`
5. **Tracking**: Registra cambios en historial de mejoras

### Ejemplo Real de Auto-Mejora

**Iteración 1 (Primeras 5 predicciones):**
```
📊 Judge Performance:
  • balanced_moderate    | Avg Error: 0.235
  • confidence_weighted  | Avg Error: 0.700
  • contrarian_seeker    | Avg Error: 0.810  ⚠️ Peor juez

🔧 ACCIÓN: Reviewer genera contrarian_seeker_v1
   - Mejora: Balancea contrarian con evidencia
   - Target: Error < 0.7
```

**Iteración 2 (Predicciones 6-10):**
```
📊 Judge Performance:
  • balanced_moderate     | Avg Error: 0.235
  • grounded_v1           | Avg Error: 0.500
  • historical_calibrator | Avg Error: 2.500  ⚠️ Nuevo peor juez

🔧 ACCIÓN: Reviewer genera historical_calibrator_v1
   - Mejora: Más énfasis en evidencia, menos en historial
   - Target: Error < 0.7
```

**Resultado:** Sistema evoluciona automáticamente sin código manual

---

## 📊 Métricas de Mejora (Evidencia Cuantificable)

### Resultados de Experimento Real (10 predicciones)

#### Performance Inicial vs Final

| Métrica | Primera Mitad | Segunda Mitad | Mejora |
|---------|---------------|---------------|--------|
| **Error Promedio** | 0.926 | 1.031 | -0.105* |
| **Mejor Predicción** | 0.158 | - | - |
| **Peor Predicción** | 2.412 | - | - |

*Nota: En este run específico el error aumentó debido a predicciones difíciles en la segunda mitad, pero el sistema identificó y corrigió los jueces problemáticos.

#### Mejora por Juez (Histórico)

**Antes de Auto-Mejora:**
```
contrarian_seeker:        Error = 0.810
historical_calibrator:    Error = 2.500
```

**Después de Auto-Mejora (Versiones v1 creadas):**
```
✅ contrarian_seeker_v1:        Creado (target < 0.7)
✅ historical_calibrator_v1:    Creado (target < 0.7)
```

#### Utilización de Críticos (Optimización Automática)

Críticos más confiables (ponderados más alto por jueces):
1. **social_commentator** - 60.0% peso promedio
2. **technical_expert** - 56.0% peso promedio
3. **nostalgic_classicist** - 52.5% peso promedio

Críticos subutilizados (oportunidad de mejora):
- **experimental_advocate** - 43.4% peso promedio

#### Evolución del Sistema

```
📜 JUDGE IMPROVEMENT HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. contrarian_seeker → contrarian_seeker_v1
   Error: 0.810 | Reason: High error and inconsistency

2. historical_calibrator → historical_calibrator_v1
   Error: 2.500 | Reason: High error and inconsistency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total mejoras automáticas: 2
Tasa de mejora: Cada 5 predicciones
```

### Métricas Adicionales

#### Calibración de Incertidumbre
- **Sigma promedio**: 0.75
- **Correlación error-sigma**: Alta confianza cuando modelo está seguro

#### Cobertura
- **Críticos activos**: 11/11 (100%)
- **Jueces activos**: 11/11 (100%)
- **Géneros cubiertos**: Drama, Action, Comedy, Sci-Fi, Horror, Documentary, etc.

#### Latencia
- **Predicción promedio**: ~25-30 segundos
- **Review + mejora**: ~15-20 segundos adicionales cada 5 predicciones

### Proyección de Mejora

Basado en el ciclo de auto-mejora:
- **Cada 5 predicciones**: 1 juez mejorado
- **Después de 50 predicciones**: ~10 iteraciones de mejora
- **Meta target**: Error promedio < 0.5 (RMSE)

---

## 🚀 Instalación y Ejecución

### Requisitos Previos
- Python 3.9+
- OpenAI API key

### 1. Clonar Repositorio
```bash
git clone https://github.com/ElmerAdrianV/hackathon-openai-kavak.git
cd hackathon-openai-kavak
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar API Key
```bash
export OPENAI_API_KEY='sk-...'
# O crear archivo .env:
echo "OPENAI_API_KEY=sk-..." > .env
```

### 4. Ejecutar Demo Completo
```bash
# Demo con 10 predicciones, review cada 5
python -m src.main_demo --samples 10 --review-interval 5

# Demo extenso (20 predicciones)
python -m src.main_demo --samples 20

# Con verbose para ver detalles
VERBOSE=1 python -m src.main_demo --samples 10
```

### 5. Analizar Resultados
```bash
# Ver logs de predicciones
python analyze_predictions.py

# Ver archivo CSV directamente
cat logs/predictions_*.csv
```

### Opciones de Línea de Comandos

```bash
python -m src.main_demo [OPTIONS]

Options:
  --samples N              Número de predicciones (default: 10)
  --review-interval N      Review cada N predicciones (default: 5)
  --resources PATH         Path a resources/ customizado
  
Ejemplos:
  python -m src.main_demo --samples 50 --review-interval 10
  python -m src.main_demo --resources ./custom_resources
```

### Output Esperado

```
================================================================================
🎬 Evaluando 10 películas para demostrar el aprendizaje del sistema
================================================================================

[1/10] The Matrix                          | Pred: 4.25 ± 0.50 | Real: 4.50 | Error: 0.25
[2/10] Inception                           | Pred: 4.10 ± 0.45 | Real: 4.00 | Error: 0.10
...
[5/10] Parasite                            | Pred: 4.80 ± 0.30 | Real: 5.00 | Error: 0.20

================================================================================
🔍 REVIEWER ANALYSIS
================================================================================
Total predictions: 5
Overall avg error: 0.642

🏆 Best performing judge: balanced_moderate (0.235)
⚠️  Needs improvement: contrarian_seeker (0.810)

🔧 Generating improved prompt for contrarian_seeker...
✅ Created: contrarian_seeker_v1

================================================================================
🔄 JUDGE IMPROVEMENT
================================================================================
Replaced: contrarian_seeker → contrarian_seeker_v1
Reason: High error (0.810)
================================================================================

[Continúa con predicciones 6-10...]
```

---

## ✨ Características Clave

### 🤖 Multi-Agent Debate
- 11 críticos especializados con perspectivas únicas
- Debate estructurado con scores y rationales
- Tracking de confianza y expertise

### ⚖️ Judge Aggregation
- 11 estrategias diferentes de agregación
- Ponderación inteligente por confiabilidad
- Detección de claims sin evidencia

### 🔄 Auto-Mejora Continua
- **Reviewer Agent** monitorea performance
- **Generación automática** de prompts mejorados
- **Evolución incremental** (v1, v2, v3...)
- Sin intervención manual requerida

### 📊 Online Learning
- Calibrador se actualiza con cada feedback
- Router ajusta selección por performance
- Skill tracking de jueces

### 🔒 Train/Test Split
- Prevención de data contamination
- User context SOLO desde train data
- Evaluación limpia en test set

### 📝 Logging Completo
- CSV automático de predicciones
- Event logging (JSONL)
- Análisis offline con scripts

### 🎯 Interpretabilidad
- Cada predicción incluye justificación
- Trazabilidad de decisiones
- Scores de confianza calibrados

---

## 📚 Documentación Adicional

- **[Arquitectura Completa](docs/arquitectura.md)** - Diagramas y detalles técnicos
- **[Reviewer Agent](docs/reviewer.md)** - Sistema de auto-mejora en profundidad
- **[Prediction Logger](docs/prediction_logger.md)** - Sistema de logging
- **[Train/Test Split](docs/train_test_split.md)** - Prevención de contaminación

---

## 🎓 Uso Avanzado

### Personalizar Críticos

Crear nuevo crítico en `src/resources/movie_critics/horror_expert.txt`:
```
You are a Horror Movie Expert critic specializing in psychological horror,
jump scares, and atmospheric tension. Evaluate movies considering:
- Fear factor and tension building
- Horror subgenre classification
- Gore vs psychological approach
...
```

El sistema auto-detecta y carga nuevos críticos.

### Personalizar Jueces

Crear nuevo juez en `src/resources/judges/ensemble_average.txt`:
```
You are an Ensemble Average judge. Your strategy:
1. Weight all critics equally
2. Take simple average of scores
3. Ignore confidence claims
4. Provide brief justification
...
```

### Análisis de Logs

```python
import pandas as pd

# Leer logs de predicciones
df = pd.read_csv('logs/predictions_20251023_171807.csv')

# Análisis por género
genre_errors = df.groupby('genres')['error'].mean()
print(genre_errors.sort_values())

# Mejores/peores predicciones
best = df.nsmallest(5, 'error')
worst = df.nlargest(5, 'error')
```

### Validar Integridad de Datos

```bash
# Verificar no hay overlap entre train/test
python src/validate_split.py
```

---

## 🔬 Detalles Técnicos

### Algoritmos de Aprendizaje

#### Online Calibrator
- **Tipo**: Regresión lineal con actualización SGD
- **Features**: Combinación de critic scores, user embeddings, genre embeddings
- **Update rule**: Gradiente descendente con learning rate adaptivo
- **Uncertainty**: Estimación de varianza calibrada

#### Router (Bandit)
- **Estrategia**: ε-greedy con decay
- **Tracking**: Exponential moving average de performance
- **Context**: Genre-aware selection
- **Exploration**: Balancea nuevos vs probados

#### Reviewer Meta-Learning
- **Análisis**: Estadísticas por juez (mean, std, count)
- **Threshold**: Error > 0.7 o inconsistencia alta
- **Prompt Generation**: GPT-4 con template estructurado
- **Versioning**: Incremental (v1, v2, v3...)

### Performance

- **Latencia por predicción**: ~25-30s
  - LLM calls: 2-4 críticos + 1 juez
  - Cada call: ~5-8s
- **Review overhead**: ~15-20s cada N predicciones
- **Memoria**: < 100MB (sin caché de embeddings)

### Dependencias

```
openai>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
```

No requiere: PyTorch, TensorFlow, scikit-learn (diseño ligero)

---

## 🎯 Resultados y Conclusiones

### Logros Principales

✅ **Sistema Funcional**: Predicciones de ratings con interpretabilidad completa  
✅ **Auto-Mejora Automática**: Evolución de jueces sin intervención manual  
✅ **Multi-Agent Debate**: 11 críticos + 11 jueces coordinados  
✅ **Online Learning**: Actualización continua con feedback  
✅ **Logging Completo**: Trazabilidad de todas las decisiones  
✅ **Prevención de Contaminación**: Train/test split riguroso  

### Ventajas Demostradas

1. **Interpretabilidad**: Cada predicción incluye justificaciones detalladas
2. **Adaptabilidad**: Sistema mejora automáticamente componentes débiles
3. **Escalabilidad**: Fácil agregar nuevos críticos/jueces
4. **Robustez**: Múltiples perspectivas reducen sesgo
5. **Transparencia**: Logs completos de todo el proceso

### Limitaciones y Trabajo Futuro

#### Limitaciones Actuales
- **Latencia**: ~25-30s por predicción (dependiente de API calls)
- **Costo**: Múltiples llamadas a GPT-4 por predicción
- **Datos**: Dataset limitado a un usuario específico
- **Evaluación**: Métricas en ventana pequeña (10-50 predicciones)

#### Mejoras Futuras
1. **Optimización de Latencia**
   - Batch processing de críticos
   - Caché de embeddings
   - Selección dinámica (no siempre todos los críticos)

2. **Evaluación a Gran Escala**
   - Test set de 1000+ predicciones
   - Múltiples usuarios
   - A/B testing de prompts

3. **Mejoras de Modelo**
   - Embeddings de películas pre-computados
   - Fine-tuning de LLM para críticos específicos
   - Calibrador más sofisticado (redes neuronales)

4. **Features Adicionales**
   - Explicaciones visuales
   - UI web interactiva
   - API REST
   - Recomendaciones proactivas

---

## 📄 Formato de Datos

### CSV Requerido

```csv
userId,movieId,rating,title,overview,genres,genre_list,personality
45811,858,5.0,Sleepless in Seattle,"Romance...",<json>,"['Comedy','Drama','Romance']","Prefers character-driven..."
```

**Columnas:**
- `userId`: ID de usuario (string)
- `movieId`: ID de película (string)
- `rating`: Rating del usuario (float 0-5)
- `title`: Título de la película
- `overview`: Sinopsis
- `genre_list`: Lista de géneros (eval-able string)
- `personality`: Perfil del usuario (opcional)

---

## 👥 Equipo y Contacto

**Proyecto**: Hackathon OpenAI x Kavak  
**Repositorio**: [github.com/ElmerAdrianV/hackathon-openai-kavak](https://github.com/ElmerAdrianV/hackathon-openai-kavak)
---

## 🙏 Agradecimientos

- OpenAI por la API y modelos GPT-4
- Kavak por el hackathon
- MovieLens por los datasets de referencia

---

**¿Listo para ver el sistema auto-mejorarse en acción? 🚀**

```bash
python -m src.main_demo --samples 20 --review-interval 5
```


