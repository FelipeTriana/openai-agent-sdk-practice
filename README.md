# openai-agent-sdk-practice

Agente RAG (*Retrieval Augmented Generation*) construido con el **OpenAI Agents SDK** y **Ollama** como backend local de LLM. Permite hacer preguntas desde la consola sobre el libro **"El agua y los sueños" de Gastón Bachelard**, usando un pipeline avanzado de recuperación con **Query Expansion** y **Re-ranking con Cross-encoder**.

**Autor:** Felipe Triana — trianagomezfelipe@gmail.com

---

## Requisitos previos

| Herramienta | Versión mínima | Cómo verificar |
|-------------|----------------|----------------|
| Python | 3.10 | `python --version` |
| Poetry | cualquiera reciente | `poetry --version` |
| Ollama | 0.17.5 | `ollama -v` |

> Ollama debe estar **en ejecución** antes de lanzar el agente. En Windows suele correr como proceso en segundo plano al iniciar sesión. Si no está activo, inícialo con `ollama serve` en otra terminal.

---

## Estructura del proyecto

```
openai-agent-sdk-practice/
├── agent.py              # Punto de entrada: loop de consola y orquestación RAG
├── docs/
│   └── Bachelard_Gaston_El_agua_y_los_suenos.pdf   # Libro fuente
├── rag/
│   ├── __init__.py       # Marca la carpeta como módulo Python
│   ├── indexer.py        # Extrae texto del PDF, genera chunks y los guarda en ChromaDB
│   └── retriever.py      # Query Expansion + búsqueda vectorial + Cross-encoder re-ranking
├── chroma_db/            # Índice vectorial generado automáticamente (no subir a git)
├── pyproject.toml        # Metadatos del proyecto y dependencias (gestionado por Poetry)
├── .env                  # Variables de entorno locales (NO subir a git)
├── .gitignore            # Archivos excluidos del repositorio
└── README.md             # Esta documentación
```

> `chroma_db/` se crea automáticamente la primera vez que ejecutas el agente. Contiene el índice vectorial del libro. No necesitas crearlo ni tocarlo manualmente.

---

## Cómo se creó este proyecto (paso a paso)

### 1. Inicializar el proyecto con Poetry

`poetry init` genera el archivo `pyproject.toml` con los metadatos del proyecto de forma interactiva. El flag `--no-interaction` omite las preguntas y crea el archivo directamente con los valores proporcionados.

```bash
poetry init --name "basic-agent" --description "Basic AI agent using OpenAI Agents SDK with Ollama" --author "Felipe Triana <trianagomezfelipe@gmail.com>" --python "^3.10" --no-interaction
```

> Esto crea el `pyproject.toml`. **No toca tu Python global** — Poetry siempre trabaja con entornos aislados.

---

### 2. Crear el entorno virtual

```bash
poetry env use python
```

> Le indica a Poetry que cree un entorno virtual aislado usando el ejecutable `python` del PATH. El entorno se almacena en el directorio de caché de Poetry (`AppData\Local\pypoetry\Cache\virtualenvs\` en Windows), completamente separado de tu instalación global.

Para confirmar dónde quedó el entorno:

```bash
poetry env info
```

---

### 3. Instalar las dependencias

Ejecutar el siguiente comando en una consola con permisos de administrador al interior de directorio.

```bash
poetry add openai-agents openai python-dotenv
```

> `poetry add` hace tres cosas a la vez:
> 1. Resuelve las versiones compatibles de cada paquete.
> 2. Las instala **dentro del entorno virtual**, nunca en el global.
> 3. Las registra automáticamente en `pyproject.toml` y genera `poetry.lock`.

| Paquete | Para qué se usa |
|---------|-----------------|
| `openai-agents` | Framework de agentes (`Agent`, `Runner`, `set_default_openai_client`, etc.) |
| `openai` | Cliente HTTP de OpenAI; se configura para apuntar a Ollama en lugar de `api.openai.com` |
| `python-dotenv` | Carga variables de entorno desde el archivo `.env` |

> **Nota:** si `poetry add` falla con errores de conexión a PyPI, corroborar que si se esté ejecutando en una terminal con permisos de administrador, si el error persite, limpiar la caché con ():
> ```bash
> poetry cache clear pypi --all
> ```
> Y vuelve a ejecutar `poetry add` en la consola con permisos de admin. Si el problema persiste, puedes instalar con pip directamente dentro del entorno de Poetry, pero esto no creará el .lock y puede afectar la puesta en marcha del entorno mas adelante generando problemas de dependencias:
> ```bash
> poetry run pip install openai-agents openai python-dotenv
> ```

---

### 4. Instalar dependencias para el agente RAG

Este segundo grupo de dependencias convierte al agente básico en un agente capaz de leer documentos, buscar información dentro de ellos y responder preguntas con contexto real. Ejecutar en una **consola con permisos de administrador**:

```bash
poetry add pymupdf chromadb sentence-transformers
```

| Paquete | Para qué se usa |
|---------|-----------------|
| `pymupdf` | Lee archivos PDF y extrae su texto página por página, preservando el orden y la estructura |
| `chromadb` | Base de datos vectorial que se ejecuta **localmente en tu máquina** (sin servidores externos); almacena los fragmentos del libro junto a sus representaciones numéricas para poder buscarlos por similitud semántica |
| `sentence-transformers` | Librería que carga y ejecuta modelos de inteligencia artificial especializados en comparar textos; es la pieza que le da "comprensión del lenguaje" al sistema de búsqueda |

#### ¿Qué son los modelos y por qué se necesitan?

Cuando instalas `sentence-transformers`, la librería por sí sola no entiende nada — es solo el motor. Los modelos son los "cerebros" que se le enchufan. Este proyecto usa dos modelos distintos para dos tareas distintas:

---

##### Modelo 1 — Bi-encoder: `all-MiniLM-L6-v2`

**¿Qué hace?**
Convierte cualquier fragmento de texto en una lista de números (llamada *embedding* o vector). La idea clave es que dos textos con significado parecido producirán vectores parecidos, aunque usen palabras distintas. Por ejemplo, *"¿Qué simboliza el río?"* y *"El significado del agua corriente"* producirán vectores muy cercanos.

**¿Para qué se usa en este proyecto?**
Se usa en dos momentos:
1. Al **indexar el libro**: cada párrafo del PDF se convierte en un vector y se guarda en ChromaDB.
2. Al **recibir una pregunta**: la pregunta del usuario también se convierte en vector, y ChromaDB busca los párrafos cuyos vectores sean más cercanos — es decir, los más relacionados semánticamente.

**¿Por qué este modelo en concreto?**
`all-MiniLM-L6-v2` es muy ligero (~90 MB) y rápido, ideal para correr en CPU sin GPU. Para un libro de filosofía/poética como El agua y los sueños de Bachelard, su capacidad de capturar similitudes semánticas es más que suficiente.

**Analogía:** imagina que cada texto es un punto en un mapa. El bi-encoder coloca textos similares cerca entre sí en ese mapa. Buscar por similitud es simplemente encontrar los puntos más cercanos a tu pregunta.

---

##### Modelo 2 — Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**¿Qué hace?**
Lee **a la vez** la pregunta del usuario y un fragmento del libro, y devuelve un número de 0 a 1 que indica qué tan relevante es ese fragmento para responder esa pregunta específica. A diferencia del bi-encoder, no trabaja con vectores separados — lee los dos textos juntos para entender la relación entre ellos.

**¿Para qué se usa en este proyecto?**
Después de que el bi-encoder recupera los ~20 fragmentos más parecidos, el cross-encoder los re-evalúa uno por uno con mayor profundidad y los reordena del más al menos relevante. Solo los 5 primeros (los más relevantes según el cross-encoder) llegan al LLM como contexto.

**¿Por qué este segundo paso?**
El bi-encoder es rápido pero aproximado — a veces sube fragmentos que son temáticamente cercanos pero no responden directamente la pregunta. El cross-encoder es más lento pero mucho más preciso, porque analiza la relación exacta entre pregunta y fragmento. Usarlos en secuencia (primero uno para filtrar, luego el otro para afinar) es la técnica conocida como **Re-ranking con Cross-encoder**.

**Analogía:** el bi-encoder es como un buscador que filtra miles de resultados en segundos usando palabras clave. El cross-encoder es como un experto que lee con calma los 20 mejores resultados y te dice cuáles realmente responden tu pregunta.

---

> **Primera ejecución — descarga automática de modelos:** la primera vez que corras el agente, `sentence-transformers` descargará los dos modelos desde HuggingFace (~170 MB en total). Necesitas conexión a internet solo esa primera vez. A partir de ahí, los modelos quedan guardados en `C:\Users\<tu_usuario>\.cache\huggingface\` y se cargan localmente sin internet.

---

### 5. Configurar las variables de entorno

Edita/crea el archivo `.env` (nunca lo subas a git — ya está en `.gitignore`):

```env
# URL base de la API compatible con OpenAI que expone Ollama localmente
OLLAMA_BASE_URL=http://localhost:11434/v1

# Ollama no requiere una API key real, pero el campo es obligatorio en el cliente OpenAI
OLLAMA_API_KEY=ollama

# Nombre del modelo a usar (debe estar descargado con: ollama pull <modelo>)
OLLAMA_MODEL=llama3.1
```

---

### 6. Descargar el modelo en Ollama

```bash
ollama pull llama3.1
```

> Para ver todos los modelos ya descargados: `ollama list`
> Puedes cambiar el modelo editando `OLLAMA_MODEL` en `.env`.

---

## Cómo acceder al entorno virtual (en futuras sesiones)

Una vez que el entorno está creado, **no necesitas recrearlo**. Para trabajar en el proyecto en el futuro, tienes dos opciones:

> **Importante:** si al reabrir el proyecto `poetry run python agent.py` falla con `ModuleNotFoundError: No module named 'agents'`, significa que el entorno virtual existe pero las dependencias no están instaladas (puede ocurrir si el entorno fue recreado o es la primera vez en una máquina nueva). Ejecuta en una **consola con permisos de administrador**:
> ```bash
> poetry install
> ```
> Si ves el error `The current project could not be installed`, es esperado — significa que Poetry intentó instalar tu proyecto como si fuera un paquete distribuible. Está controlado con `package-mode = false` en `pyproject.toml`, que le indica a Poetry que este proyecto es una aplicación/script, no una librería para publicar en PyPI.



### Opción 1 — Ejecutar comandos sin activar (recomendado para scripts)

```bash
poetry run python agent.py
```

> `poetry run` automáticamente activa el entorno, ejecuta el comando dentro de él, y luego lo desactiva. Ideal para correr scripts rápidamente.

### Opción 2 — Activar una shell interactiva dentro del entorno

```bash
poetry shell
```

> Abre una terminal dentro del entorno virtual. A partir de ahí puedes ejecutar comandos directamente sin anteponer `poetry run`:
> ```bash
> python agent.py
> poetry add nuevo-paquete
> python -c "import openai; print(openai.__version__)"
> ```
> Para salir del entorno interactivo:
> ```bash
> exit
> ```

---

## Cómo ejecutar el agente

```bash
poetry run python agent.py
```

> `poetry run` activa automáticamente el entorno virtual antes de ejecutar el script. No hace falta correr `poetry shell` manualmente.

**Primera ejecución:** el agente procesará el PDF completo y construirá el índice vectorial en `chroma_db/`. Esto puede tardar varios minutos dependiendo del tamaño del libro y la velocidad de tu máquina. Las ejecuciones siguientes cargan el índice directamente desde disco y son inmediatas.

**Uso interactivo:**

```
════════════════════════════════════════════════════════════
  Agente RAG — El agua y los sueños · Gastón Bachelard
  Escribe tu pregunta o 'salir' para terminar.
════════════════════════════════════════════════════════════

Tu pregunta: ¿Qué simboliza el agua para Bachelard?

Procesando...

Respuesta:
Para Bachelard, el agua es ante todo una materia que invita a soñar...

────────────────────────────────────────────────────────────

Tu pregunta: salir
Hasta luego.
```

---

## Cómo funciona el agente

### Pipeline RAG completo

Cada vez que escribes una pregunta, el agente ejecuta este pipeline antes de responder:

```
Tu pregunta
     │
     ▼
┌─────────────────────┐
│   Query Expansion   │  El LLM genera 4 reformulaciones de tu pregunta
└──────┬──────────────┘
       │ 5 queries (original + 4 variaciones)
       ▼
┌─────────────────────┐
│  Bi-encoder search  │  ChromaDB recupera 10 fragmentos por query → ~50 candidatos
└──────┬──────────────┘
       │ deduplicación → ~20-30 fragmentos únicos
       ▼
┌─────────────────────┐
│  Cross-encoder      │  Puntúa cada fragmento contra tu pregunta ORIGINAL
│  Re-ranking         │  y los reordena de mayor a menor relevancia
└──────┬──────────────┘
       │ top-5 fragmentos más relevantes
       ▼
┌─────────────────────┐
│  LLM (llama3.1)     │  Recibe la pregunta + los 5 fragmentos como contexto
│                     │  y genera la respuesta final
└─────────────────────┘
```

### Módulos del proyecto

#### `rag/indexer.py` — Construcción del índice

Se ejecuta automáticamente la primera vez y solo una vez:

1. **Extrae el texto** del PDF página por página con `pymupdf`.
2. **Divide el texto en fragmentos** (*chunks*) de 400 palabras con 50 palabras de superposición entre fragmentos consecutivos. La superposición evita que una idea quede cortada entre dos chunks sin contexto.
3. **Genera embeddings** de cada fragmento con el bi-encoder `all-MiniLM-L6-v2`.
4. **Guarda los fragmentos y sus embeddings** en ChromaDB de forma persistente en `chroma_db/`.

#### `rag/retriever.py` — Recuperación con Query Expansion + Re-ranking

Ejecuta el pipeline completo por cada pregunta:

1. **Query Expansion:** el LLM genera 4 reformulaciones de tu pregunta original. Esto es clave porque el libro puede hablar del mismo concepto con vocabulario diferente al que tú usas.
2. **Bi-encoder retrieval:** para cada una de las 5 queries (original + 4 variaciones), ChromaDB convierte la query en vector y busca los 10 fragmentos más cercanos. Resultado: hasta 50 candidatos que se dedupliccan.
3. **Cross-encoder re-ranking:** el cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` lee cada candidato junto a la pregunta ORIGINAL y asigna un puntaje de relevancia. Se ordenan de mayor a menor y solo los 5 mejores pasan al LLM.

#### `agent.py` — Orquestador y loop de consola

- Configura el cliente de Ollama y lo registra en el SDK.
- Llama a `build_index()` para tener el índice listo.
- Entra en un loop que lee preguntas del usuario, llama a `retrieve()`, construye el prompt con el contexto recuperado, invoca al agente y muestra la respuesta.

### Por qué `asyncio.run(main())`

`Runner.run(...)` es una operación asíncrona (usa `await`). En Python, `await` solo se puede usar dentro de una función `async` y necesita un *event loop* activo.

- `async def main()`: define una corrutina donde podemos usar `await`.
- `asyncio.run(main())`: crea y gestiona el *event loop*, ejecuta `main` hasta que termina, y lo cierra.

| Concepto | Descripción |
|----------|-------------|
| `Agent` | Define el agente: nombre, instrucciones (system prompt) y modelo. |
| `instructions` | System prompt: le dice al LLM que responda solo con el contexto del libro. |
| `Runner.run()` | Ejecuta el agente de forma asíncrona con el prompt enriquecido con contexto. |
| `final_output` | Texto final generado por el LLM. |
| `build_index()` | Procesa el PDF y crea el índice vectorial (solo la primera vez). |
| `retrieve()` | Pipeline Query Expansion + bi-encoder + cross-encoder re-ranking. |

---

## Solución de problemas

| Error | Causa probable | Solución |
|-------|----------------|----------|
| `Connection refused` en `localhost:11434` | Ollama no está ejecutándose | Ejecuta `ollama serve` |
| `model not found` | El modelo no está descargado | Ejecuta `ollama pull llama3.1` |
| `No module named 'agents'` | Dependencias no instaladas | Ejecuta `poetry install` con permisos de administrador |
| `No module named 'rag'` | Ejecutas Python directamente sin Poetry | Usa siempre `poetry run python agent.py` |
| `All attempts to connect to pypi.org failed` | Caché de Poetry corrupta | Borra la caché (ver paso 3) y reintenta |
| `FileNotFoundError: docs/Bachelard...pdf` | El PDF no está en la carpeta `docs/` | Coloca el PDF en `docs/` con el nombre exacto |
| El índice tarda mucho la primera vez | Comportamiento normal | Solo ocurre una vez; las siguientes ejecuciones son rápidas |
