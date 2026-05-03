from pathlib import Path
import re
import shutil
import unicodedata
from collections import defaultdict

import pandas as pd


# Ruta del archivo CSV original entregado por la tarea.
RAW_DATA_PATH = Path("data/noticias_chile_2023_2025.csv")

# Carpeta donde se guardará el Data Warehouse generado.
WAREHOUSE_PATH = Path("warehouse")

# Cantidad de filas que se procesan por bloque.
# Esto evita cargar millones de noticias completas en memoria al mismo tiempo.
CHUNK_SIZE = 250_000


# Diccionario de regiones y alias.
# Cada región tiene palabras o frases que pueden aparecer en el título o cuerpo de una noticia.
# Estos alias permiten inferir a qué región se refiere cada artículo.
REGION_ALIASES = {
    "Arica y Parinacota": [
        "arica", "parinacota", "putre", "camarones", "general lagos"
    ],
    "Tarapacá": [
        "tarapaca", "iquique", "alto hospicio", "pozo almonte", "pica", "huara"
    ],
    "Antofagasta": [
        "antofagasta", "calama", "tocopilla", "mejillones", "taltal", "san pedro de atacama"
    ],
    "Atacama": [
        "atacama", "copiapo", "caldera", "vallenar", "chañaral", "diego de almagro", "huasco"
    ],
    "Coquimbo": [
        "coquimbo", "la serena", "ovalle", "illapel", "vicuna", "vicuña", "los vilos", "salamanca"
    ],
    "Valparaíso": [
        "valparaiso", "valpo", "vina del mar", "viña del mar", "quilpue", "quilpué",
        "villa alemana", "san antonio", "quillota", "san felipe", "isla de pascua"
    ],
    "Metropolitana": [
    "metropolitana", "region metropolitana", "rm", "santiago", "providencia",
    "las condes", "maipu", "maipú", "puente alto", "la florida", "ñuñoa",
    "nunoa", "la reina", "vitacura", "lo barnechea", "estacion central",
    "estación central", "san bernardo"
    ],
    "O'Higgins": [
    "ohiggins", "o higgins", "o'higgins",
    "rancagua", "san fernando", "pichilemu", "machali", "machalí",
    "rengo", "san vicente", "san vicente de tagua tagua", "coltauco",
    "graneros", "requinoa", "requínoa", "mostazal", "peumo", "chimbarongo"
    ],
    "Maule": [
        "maule", "talca", "curico", "curicó", "linares", "cauquenes", "constitucion", "constitución"
    ],
    "Ñuble": [
        "nuble", "ñuble", "chillan", "chillán", "san carlos", "bulnes", "quirihue"
    ],
    "Biobío": [
        "biobio", "bíobío", "bio bio", "bío bío", "concepcion", "concepción",
        "talcahuano", "los angeles", "los ángeles", "coronel", "lota", "hualpen", "hualpén"
    ],
    "La Araucanía": [
        "araucania", "araucanía", "temuco", "villarrica", "pucon", "pucón", "angol", "padre las casas"
    ],
    "Los Ríos": [
        "los rios", "los ríos", "valdivia", "la union", "la unión", "rio bueno", "río bueno", "panguipulli"
    ],
    "Los Lagos": [
        "los lagos", "puerto montt", "osorno", "chiloe", "chiloé", "castro", "ancud", "frutillar"
    ],
    "Aysén": [
        "aysen", "aysén", "coyhaique", "coihaique", "puerto aysen", "puerto aysén"
    ],
    "Magallanes": [
        "magallanes", "punta arenas", "puerto natales", "porvenir", "tierra del fuego", "antartica", "antártica"
    ],
}


def normalize_text(value):
    # Convierte valores nulos en texto vacío para evitar errores.
    if pd.isna(value):
        return ""

    # Pasa todo a minúsculas para comparar sin depender de mayúsculas o minúsculas.
    value = str(value).lower()

    # Normaliza caracteres Unicode para poder quitar tildes.
    value = unicodedata.normalize("NFD", value)

    # Elimina marcas de acento, por ejemplo transforma "Biobío" en "Biobio".
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")

    # Normaliza comillas raras para evitar problemas con nombres como O'Higgins.
    value = value.replace("’", "'").replace("`", "'").replace("´", "'")

    # Reemplaza símbolos extraños por espacios, manteniendo letras, números, ñ y apóstrofes.
    value = re.sub(r"[^a-z0-9ñ\s']", " ", value)

    # Elimina espacios repetidos.
    value = re.sub(r"\s+", " ", value).strip()

    return value


def count_words(value):
    # Normaliza el texto antes de contar palabras.
    text = normalize_text(value)

    # Si el texto queda vacío, el conteo es cero.
    if text == "":
        return 0

    # Extrae tokens tipo palabra.
    words = re.findall(r"\b[a-z0-9ñ']+\b", text)

    return len(words)


def build_region_patterns(region_aliases):
    # Construye expresiones regulares para cada región.
    # Esto permite buscar alias regionales en título y cuerpo.
    patterns = {}

    for region, aliases in region_aliases.items():
        # Normaliza cada alias para que la búsqueda sea insensible a tildes.
        normalized_aliases = [normalize_text(alias) for alias in aliases]

        # Escapa caracteres especiales de regex y permite espacios flexibles.
        escaped_aliases = [
            re.escape(alias).replace("\\ ", r"\s+")
            for alias in normalized_aliases
            if alias
        ]

        # Usa límites de palabra para evitar coincidencias falsas dentro de palabras más largas.
        pattern = r"\b(" + "|".join(escaped_aliases) + r")\b"

        # Guarda el patrón compilado para usarlo muchas veces de forma más eficiente.
        patterns[region] = re.compile(pattern, flags=re.IGNORECASE)

    return patterns


# Patrones globales de regiones.
# Se construyen una sola vez para no recompilarlos por cada noticia.
REGION_PATTERNS = build_region_patterns(REGION_ALIASES)


def infer_region(title, body):
    # Normaliza título y cuerpo para buscar coincidencias sin depender de tildes o mayúsculas.
    title_norm = normalize_text(title)
    body_norm = normalize_text(body)

    # Guarda cuántas veces aparece cada región en el título y en el cuerpo.
    title_counts = {}
    body_counts = {}

    for region, pattern in REGION_PATTERNS.items():
        title_counts[region] = len(pattern.findall(title_norm))
        body_counts[region] = len(pattern.findall(body_norm))

    # Guarda solo las regiones que sí aparecieron al menos una vez.
    candidates = []

    for region in REGION_PATTERNS:
        title_mentions = title_counts[region]
        body_mentions = body_counts[region]
        total_mentions = title_mentions + body_mentions

        if total_mentions > 0:
            candidates.append(
                {
                    "region_name": region,
                    "title_mentions": title_mentions,
                    "body_mentions": body_mentions,
                    "total_mentions": total_mentions,
                }
            )

    # Si no aparece ninguna región, se asigna la categoría especial Desconocida.
    if not candidates:
        return "Desconocida", 0, 0, 0

    # Regla de desambiguación:
    # se prioriza la región con más menciones en el título,
    # luego la de mayor número total de menciones,
    # luego la de más menciones en el cuerpo.
    candidates = sorted(
        candidates,
        key=lambda x: (
            x["title_mentions"],
            x["total_mentions"],
            x["body_mentions"],
            x["region_name"]
        ),
        reverse=True
    )

    winner = candidates[0]

    return (
        winner["region_name"],
        winner["title_mentions"],
        winner["body_mentions"],
        winner["total_mentions"],
    )


def clean_chunk(df):
    # Columnas que deben existir en el CSV crudo según el enunciado.
    expected_columns = [
        "article_id",
        "title",
        "body",
        "publish_date",
        "source",
        "country",
    ]

    # Revisa si falta alguna columna obligatoria.
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Faltan columnas en el CSV crudo: {missing_columns}")

    # Conserva solo las columnas relevantes y evita modificar directamente el chunk original.
    df = df[expected_columns].copy()

    # Limpia identificadores, títulos, cuerpos, fuentes y país.
    df["article_id"] = df["article_id"].astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["body"] = df["body"].fillna("").astype(str).str.strip()
    df["source"] = df["source"].fillna("desconocido").astype(str).str.strip().str.lower()
    df["country"] = df["country"].fillna("Chile").astype(str).str.strip()

    # Convierte publish_date a fecha.
    # Los valores inválidos se convierten en NaT.
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

    # Elimina noticias sin fecha válida porque no se podrían particionar por año y mes.
    df = df.dropna(subset=["publish_date"])

    # Elimina duplicados dentro del chunk usando article_id.
    df = df.drop_duplicates(subset=["article_id"])

    # Extrae atributos temporales útiles para la fact table y particionamiento.
    df["year"] = df["publish_date"].dt.year.astype(int)
    df["month"] = df["publish_date"].dt.month.astype(int)
    df["day"] = df["publish_date"].dt.day.astype(int)

    # Calcula métricas simples del texto.
    # Estas métricas son hechos derivados que quedan en fact_news.
    df["title_word_count"] = df["title"].apply(count_words)
    df["body_word_count"] = df["body"].apply(count_words)
    df["total_word_count"] = df["title_word_count"] + df["body_word_count"]

    # Aplica la inferencia regional a cada noticia.
    region_results = df.apply(
        lambda row: infer_region(row["title"], row["body"]),
        axis=1,
        result_type="expand"
    )

    # Nombra las columnas generadas por la inferencia regional.
    region_results.columns = [
        "region_name",
        "region_title_mentions",
        "region_body_mentions",
        "region_total_mentions",
    ]

    # Une los resultados regionales al dataframe limpio.
    df = pd.concat([df, region_results], axis=1)

    return df


def first_pass_collect_dimensions():
    # Esta primera pasada ahora es liviana.
    # Solo recolecta fechas y fuentes, sin leer title ni body.
    # Esto evita que el Paso 1 se quede demasiado tiempo procesando texto pesado.

    dates = set()
    sources = set()
    raw_rows_after_cleaning = 0

    # Columnas mínimas necesarias para construir dim_date y dim_source.
    usecols = [
        "article_id",
        "publish_date",
        "source",
    ]

    # Lee el CSV por chunks para no cargar todo el dataset en RAM.
    for chunk_number, chunk in enumerate(
        pd.read_csv(RAW_DATA_PATH, usecols=usecols, chunksize=CHUNK_SIZE)
    ):
        # Limpia article_id para detectar duplicados dentro del chunk.
        chunk["article_id"] = chunk["article_id"].astype(str).str.strip()

        # Normaliza source para que un mismo medio no aparezca duplicado por mayúsculas o espacios.
        chunk["source"] = chunk["source"].fillna("desconocido").astype(str).str.strip().str.lower()

        # Convierte publish_date a formato fecha.
        chunk["publish_date"] = pd.to_datetime(chunk["publish_date"], errors="coerce")

        # Elimina filas sin fecha válida porque no sirven para la dimensión de fecha ni para particionar.
        chunk = chunk.dropna(subset=["publish_date"])

        # Elimina duplicados dentro del chunk.
        chunk = chunk.drop_duplicates(subset=["article_id"])

        # Suma el número de filas que pasarán al warehouse según esta limpieza básica.
        raw_rows_after_cleaning += len(chunk)

        # Guarda fechas únicas para construir dim_date.
        dates.update(chunk["publish_date"].dt.date.astype(str).unique())

        # Guarda fuentes únicas para construir dim_source.
        sources.update(chunk["source"].unique())

        # Mensaje de progreso para saber que el programa no está congelado.
        print(f"Paso 1 - chunk {chunk_number} procesado: {len(chunk)} filas")

    # Las regiones no necesitan descubrirse desde los datos en esta primera pasada.
    # Se construyen directamente desde la tabla de alias definida manualmente.
    regions = set(REGION_ALIASES.keys())

    # Se agrega la región especial pedida para artículos sin coincidencia.
    regions.add("Desconocida")

    return dates, sources, regions, raw_rows_after_cleaning


def build_dim_date(dates):
    # Construye la dimensión temporal a partir de todas las fechas únicas encontradas.
    dim_date = pd.DataFrame({"date": sorted(dates)})

    # Convierte la columna date a tipo fecha.
    dim_date["date"] = pd.to_datetime(dim_date["date"])

    # Crea una llave subrogada para la dimensión.
    dim_date["date_sk"] = range(1, len(dim_date) + 1)

    # Extrae atributos útiles para análisis temporal.
    dim_date["year"] = dim_date["date"].dt.year.astype(int)
    dim_date["month"] = dim_date["date"].dt.month.astype(int)
    dim_date["day"] = dim_date["date"].dt.day.astype(int)
    dim_date["day_of_week"] = dim_date["date"].dt.day_name()
    dim_date["week_of_year"] = dim_date["date"].dt.isocalendar().week.astype(int)
    dim_date["quarter"] = dim_date["date"].dt.quarter.astype(int)
    dim_date["year_month"] = dim_date["date"].dt.strftime("%Y-%m")

    # Ordena las columnas para que la llave quede primero.
    dim_date = dim_date[
        [
            "date_sk",
            "date",
            "year",
            "month",
            "day",
            "day_of_week",
            "week_of_year",
            "quarter",
            "year_month",
        ]
    ]

    return dim_date


def build_dim_source(sources):
    # Construye la dimensión de fuentes o medios de comunicación.
    dim_source = pd.DataFrame({"source": sorted(sources)})

    # Crea una llave subrogada para cada medio.
    dim_source["source_sk"] = range(1, len(dim_source) + 1)

    # Ordena columnas.
    dim_source = dim_source[
        [
            "source_sk",
            "source",
        ]
    ]

    return dim_source


def build_dim_region(regions):
    # Construye la dimensión de regiones.
    # También guarda los alias usados para justificar trazabilidad del ETL.
    rows = []

    for region in sorted(regions):
        aliases = REGION_ALIASES.get(region, [])

        rows.append(
            {
                "region_name": region,
                "region_aliases": "|".join(aliases),
                "is_known_region": region != "Desconocida",
            }
        )

    dim_region = pd.DataFrame(rows)

    # Crea una llave subrogada para cada región.
    dim_region["region_sk"] = range(1, len(dim_region) + 1)

    # Ordena columnas.
    dim_region = dim_region[
        [
            "region_sk",
            "region_name",
            "region_aliases",
            "is_known_region",
        ]
    ]

    return dim_region


def save_dimensions(dim_date, dim_source, dim_region):
    # Crea las carpetas de dimensiones dentro del warehouse.
    (WAREHOUSE_PATH / "dim_date").mkdir(parents=True, exist_ok=True)
    (WAREHOUSE_PATH / "dim_source").mkdir(parents=True, exist_ok=True)
    (WAREHOUSE_PATH / "dim_region").mkdir(parents=True, exist_ok=True)

    # Guarda las dimensiones en formato Parquet.
    # Parquet es adecuado para analítica porque es columnar.
    dim_date.to_parquet(WAREHOUSE_PATH / "dim_date" / "dim_date.parquet", index=False)
    dim_source.to_parquet(WAREHOUSE_PATH / "dim_source" / "dim_source.parquet", index=False)
    dim_region.to_parquet(WAREHOUSE_PATH / "dim_region" / "dim_region.parquet", index=False)


def second_pass_build_fact(dim_date, dim_source, dim_region):
    # Construye diccionarios para traducir valores reales a llaves subrogadas.
    date_map = {
        row["date"].date().isoformat(): row["date_sk"]
        for _, row in dim_date.iterrows()
    }

    source_map = {
        row["source"]: row["source_sk"]
        for _, row in dim_source.iterrows()
    }

    region_map = {
        row["region_name"]: row["region_sk"]
        for _, row in dim_region.iterrows()
    }

    # Crea la carpeta base de la tabla de hechos.
    fact_base_path = WAREHOUSE_PATH / "fact_news"
    fact_base_path.mkdir(parents=True, exist_ok=True)

    # Lleva la cuenta de cuántos archivos se han escrito por partición.
    partition_counters = defaultdict(int)

    # Lee nuevamente el CSV, ahora completo, porque se necesita title y body para la fact table.
    for chunk_number, chunk in enumerate(pd.read_csv(RAW_DATA_PATH, chunksize=CHUNK_SIZE)):
        # Limpia el chunk, calcula conteos de palabras e infiere región.
        clean = clean_chunk(chunk)

        # Crea una versión string de la fecha para buscar su date_sk.
        clean["date_key_str"] = clean["publish_date"].dt.date.astype(str)

        # Asigna llaves subrogadas desde las dimensiones.
        clean["date_sk"] = clean["date_key_str"].map(date_map)
        clean["source_sk"] = clean["source"].map(source_map)
        clean["region_sk"] = clean["region_name"].map(region_map)

        # Selecciona las columnas finales de la fact table.
        fact_news = clean[
            [
                "article_id",
                "date_sk",
                "source_sk",
                "region_sk",
                "title",
                "body",
                "country",
                "publish_date",
                "year",
                "month",
                "day",
                "title_word_count",
                "body_word_count",
                "total_word_count",
                "region_title_mentions",
                "region_body_mentions",
                "region_total_mentions",
            ]
        ].copy()

        # Escribe cada grupo de noticias en la carpeta correspondiente a su año y mes.
        for (year, month), group in fact_news.groupby(["year", "month"]):
            partition_path = fact_base_path / f"year={year}" / f"month={month:02d}"
            partition_path.mkdir(parents=True, exist_ok=True)

            part_number = partition_counters[(year, month)]
            output_file = partition_path / f"part-{part_number:05d}.parquet"

            group.to_parquet(output_file, index=False)

            partition_counters[(year, month)] += 1

        # Mensaje de progreso para saber por qué chunk va el ETL.
        print(f"Paso 4 - chunk {chunk_number} procesado: {len(clean)} artículos")


def read_fact_news():
    # Busca todos los archivos Parquet de la fact table.
    fact_path = WAREHOUSE_PATH / "fact_news"
    files = list(fact_path.glob("year=*/month=*/*.parquet"))

    # Si no hay archivos, significa que la fact table no se generó.
    if not files:
        raise ValueError("No se encontraron archivos Parquet en fact_news.")

    # Lee todos los Parquet y los une para poder validar.
    return pd.concat(
        [pd.read_parquet(file) for file in files],
        ignore_index=True
    )


def validate_warehouse(expected_fact_rows):
    # Lee dimensiones desde el warehouse.
    dim_date = pd.read_parquet(WAREHOUSE_PATH / "dim_date" / "dim_date.parquet")
    dim_source = pd.read_parquet(WAREHOUSE_PATH / "dim_source" / "dim_source.parquet")
    dim_region = pd.read_parquet(WAREHOUSE_PATH / "dim_region" / "dim_region.parquet")

    # Lee la tabla de hechos completa para validar consistencia.
    fact_news = read_fact_news()

    # Lista donde se acumulan errores de validación.
    errors = []

    # Revisa consistencia referencial.
    # Es decir, ninguna llave en fact_news debe apuntar a una dimensión inexistente.
    orphan_dates = set(fact_news["date_sk"]) - set(dim_date["date_sk"])
    orphan_sources = set(fact_news["source_sk"]) - set(dim_source["source_sk"])
    orphan_regions = set(fact_news["region_sk"]) - set(dim_region["region_sk"])

    if orphan_dates:
        errors.append(f"Hay date_sk huérfanas: {orphan_dates}")

    if orphan_sources:
        errors.append(f"Hay source_sk huérfanas: {orphan_sources}")

    if orphan_regions:
        errors.append(f"Hay region_sk huérfanas: {orphan_regions}")

    # Revisa que el número de filas de fact_news coincida con lo esperado.
    if len(fact_news) != expected_fact_rows:
        errors.append(
            f"Conteo incorrecto en fact_news: esperado={expected_fact_rows}, obtenido={len(fact_news)}"
        )

    # Revisa que las llaves subrogadas de las dimensiones no estén duplicadas.
    if dim_date["date_sk"].duplicated().any():
        errors.append("Hay date_sk duplicadas en dim_date.")

    if dim_source["source_sk"].duplicated().any():
        errors.append("Hay source_sk duplicadas en dim_source.")

    if dim_region["region_sk"].duplicated().any():
        errors.append("Hay region_sk duplicadas en dim_region.")

    # Revisa que los valores naturales de las dimensiones tampoco estén duplicados.
    if dim_date["date"].duplicated().any():
        errors.append("Hay fechas duplicadas en dim_date.")

    if dim_source["source"].duplicated().any():
        errors.append("Hay sources duplicados en dim_source.")

    if dim_region["region_name"].duplicated().any():
        errors.append("Hay regiones duplicadas en dim_region.")

    # Revisa que cada archivo esté guardado en la partición correcta.
    wrong_partitions = []

    for file in (WAREHOUSE_PATH / "fact_news").glob("year=*/month=*/*.parquet"):
        file_df = pd.read_parquet(file)

        year_from_folder = int(file.parts[-3].replace("year=", ""))
        month_from_folder = int(file.parts[-2].replace("month=", ""))

        invalid_rows = file_df[
            (file_df["year"] != year_from_folder)
            | (file_df["month"] != month_from_folder)
        ]

        if len(invalid_rows) > 0:
            wrong_partitions.append(str(file))

    if wrong_partitions:
        errors.append(f"Hay archivos con particiones incorrectas: {wrong_partitions[:5]}")

    # Revisa que el conteo total de palabras sea consistente.
    invalid_word_counts = fact_news[
        fact_news["total_word_count"]
        != fact_news["title_word_count"] + fact_news["body_word_count"]
    ]

    if len(invalid_word_counts) > 0:
        errors.append("Hay filas donde total_word_count no coincide con title + body.")

    # Revisa que no existan llaves nulas en la fact table.
    null_keys = fact_news[
        fact_news[["date_sk", "source_sk", "region_sk"]].isna().any(axis=1)
    ]

    if len(null_keys) > 0:
        errors.append("Hay filas en fact_news con llaves nulas.")

    # Si hay errores, se imprimen y se detiene el programa.
    if errors:
        print("\nVALIDACIÓN FALLIDA")

        for error in errors:
            print(f"- {error}")

        raise ValueError("El warehouse no pasó las validaciones.")

    # Si no hay errores, se imprime un resumen de validación exitosa.
    print("\nVALIDACIÓN EXITOSA")
    print(f"- fact_news: {len(fact_news)} filas")
    print(f"- dim_date: {len(dim_date)} fechas")
    print(f"- dim_source: {len(dim_source)} medios")
    print(f"- dim_region: {len(dim_region)} regiones")
    print("- No hay llaves foráneas huérfanas")
    print("- No hay llaves duplicadas en dimensiones")
    print("- Las particiones year/month son correctas")
    print("- Los conteos de palabras son consistentes")


def run_etl():
    # Verifica que el archivo crudo exista antes de comenzar.
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo: {RAW_DATA_PATH}")

    # Si ya existía un warehouse anterior, lo elimina para generar uno nuevo desde cero.
    if WAREHOUSE_PATH.exists():
        shutil.rmtree(WAREHOUSE_PATH)

    # Crea la carpeta principal del warehouse.
    WAREHOUSE_PATH.mkdir(parents=True, exist_ok=True)

    # Primera pasada liviana para recolectar fechas y fuentes.
    print("Paso 1: recolectando valores para dimensiones...")
    dates, sources, regions, expected_fact_rows = first_pass_collect_dimensions()

    # Construye las dimensiones del esquema estrella.
    print("Paso 2: construyendo dimensiones...")
    dim_date = build_dim_date(dates)
    dim_source = build_dim_source(sources)
    dim_region = build_dim_region(regions)

    # Guarda dimensiones en formato Parquet.
    print("Paso 3: guardando dimensiones...")
    save_dimensions(dim_date, dim_source, dim_region)

    # Segunda pasada completa para generar fact_news particionada.
    print("Paso 4: construyendo fact_news particionada...")
    second_pass_build_fact(dim_date, dim_source, dim_region)

    # Ejecuta validaciones de consistencia del warehouse.
    print("Paso 5: validando warehouse...")
    validate_warehouse(expected_fact_rows)

    # Mensaje final si todo salió bien.
    print("\nETL terminado correctamente.")


# Permite ejecutar el ETL completo al correr este archivo desde terminal.
if __name__ == "__main__":
    run_etl()