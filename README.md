# basic-agent

Agente de IA minimalista construido con el **OpenAI Agents SDK** que utiliza **Ollama** como backend local de LLM. Al ejecutarlo, el agente recibe como instrucción contar un chiste y responde en la terminal.

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
basic-agent/
├── agent.py          # Código principal del agente
├── pyproject.toml    # Metadatos del proyecto y dependencias (generado por Poetry)
├── .env              # Variables de entorno locales (NO subir a git)
├── .gitignore        # Archivos excluidos del repositorio
└── README.md         # Esta documentación
```

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

### 4. Configurar las variables de entorno

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

### 5. Descargar el modelo en Ollama

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

**Salida esperada** (el chiste varía en cada ejecución):

```
¿Por qué los pájaros vuelan hacia el sur en invierno?
¡Porque caminar sería demasiado lejos!
```

---

## Cómo funciona el agente

```python
load_dotenv()  # Carga variables de .env en os.environ

client = AsyncOpenAI(
  base_url=os.environ["OLLAMA_BASE_URL"],
  api_key=os.environ["OLLAMA_API_KEY"],
)

set_default_openai_client(client)  # El SDK usará este cliente en lugar de api.openai.com
set_tracing_disabled(True)         # Evita que el SDK intente enviar trazas a OpenAI

agent = Agent(
    name="Joke Teller",
    instructions="Eres un comediante...",  # System prompt del agente
  model=os.environ["OLLAMA_MODEL"],
)

async def main() -> None:
  result = await Runner.run(agent, input="Hola, cuentame un chiste.")
  print(result.final_output)

if __name__ == "__main__":
  asyncio.run(main())
```

### Por que `asyncio.run(main())`

`Runner.run(...)` es una operacion asincrona (usa `await`). En Python, `await` solo se puede usar dentro de una funcion `async` y necesita un *event loop* activo.

- `async def main()`: define una corrutina. Es la funcion donde podemos usar `await Runner.run(...)`.
- `asyncio.run(main())`: crea y administra el *event loop*, ejecuta la corrutina `main` hasta terminar, y cierra el loop al final.

Sin `asyncio.run(main())`, la corrutina no se ejecuta y Python no sabria como manejar el `await` en un script normal.

### Por que `main` es `async`

La llamada al modelo es I/O de red (HTTP hacia Ollama) y el SDK la expone como asincrona. `main` debe ser `async` para poder hacer:

```python
result = await Runner.run(...)
```

Si `main` no fuera `async`, esa linea daria error de sintaxis/ejecucion porque `await` no esta permitido fuera de funciones asincronas.

### Que significa `-> None`

`-> None` es una anotacion de tipo que indica que `main` no retorna ningun valor util.

- `main` imprime en pantalla el resultado (`print(...)`).
- No hace `return algo`.
- Por eso su tipo de retorno correcto es `None`.

Es documentacion para humanos y para herramientas de analisis de tipos; no cambia el comportamiento en runtime.

| Concepto | Descripción |
|----------|-------------|
| `Agent` | Define el agente: nombre, instrucciones (system prompt) y modelo. |
| `instructions` | Texto enviado como mensaje de sistema en cada llamada al LLM. |
| `Runner.run()` | Ejecuta el agente de forma asíncrona con el mensaje de usuario. |
| `final_output` | Texto final generado por el LLM. |

---

## Solución de problemas

| Error | Causa probable | Solución |
|-------|----------------|----------|
| `Connection refused` en `localhost:11434` | Ollama no está ejecutándose | Ejecuta `ollama serve` |
| `model not found` | El modelo no está descargado | Ejecuta `ollama pull llama3.1` |
| `No module named 'agents'` | Dependencias no instaladas | Ejecuta `poetry add openai-agents openai python-dotenv` |
| `All attempts to connect to pypi.org failed` | Caché de Poetry corrupta | Borra la caché (ver paso 3) y reintenta |
