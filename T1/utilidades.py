"""
utils_mapreduce.py
==================
Utilidades comunes para la Parte 2 - Analítica MapReduce.
Incluye: lector de warehouse, tokenizador, stopwords en español,
normalización de texto y funciones map/reduce base.
"""

import os
import re
import unicodedata
import functools
import itertools
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa


# ---------------------------------------------------------------------------
# STOPWORDS en español (dominio noticias)
# ---------------------------------------------------------------------------
STOPWORDS = {
    # artículos y determinantes
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # preposiciones
    "a", "ante", "bajo", "con", "contra", "de", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por",
    "segun", "sin", "sobre", "tras",
    # conjunciones
    "y", "e", "ni", "o", "u", "pero", "sino", "aunque", "porque",
    "que", "si", "como", "cuando", "donde", "mientras", "pues",
    # pronombres
    "yo", "tu", "el", "ella", "nosotros", "vosotros", "ellos", "ellas",
    "me", "te", "se", "le", "lo", "la", "nos", "os", "les",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas",
    # verbos auxiliares y comunes
    "es", "son", "era", "eran", "fue", "fueron", "ser", "estar",
    "ha", "han", "hay", "haber", "tiene", "tienen", "tener",
    "dijo", "dijo", "dice", "deben", "puede", "pueden",
    # adverbios comunes
    "no", "si", "mas", "muy", "bien", "mal", "ya", "aun", "tambien",
    "solo", "ademas", "asi", "entonces", "luego", "antes", "despues",
    "alli", "aqui", "aca",
    # otros frecuentes en noticias
    "según", "segun", "tras", "durante", "mediante", "respecto",
    "chile", "chilena", "chileno", "chilenas", "chilenos",
    # contracciones y formas compuestas
    "del", "al",
    # determinantes y cuantificadores
    "sus", "todo", "toda", "todos", "todas", "cada", "otro", "otra",
    "otros", "otras", "mismo", "misma", "mismos", "mismas",
    "gran", "grandes", "nuevo", "nueva", "nuevos", "nuevas",
    "primer", "primera", "segundo", "segunda", "ultimo", "ultima",
    # verbos comunes sin carga informativa
    "fue", "ser", "sido", "han", "haber", "hace", "hacer", "hecho",
    "habia", "habra", "tiene", "tenia", "tener",
    # adverbios y conectores
    "mas", "menos", "vez", "cual", "cuya", "cuyo",
    "ayer", "hoy", "dia", "dias", "ano", "anos", "mes", "meses",
    "mayor", "menor", "parte",
    # numerales frecuentes
    "dos", "tres", "cuatro", "cinco",
    # palabras de relleno digital
    "https", "http", "www", "com", "cl", "html", "php",
}


# ---------------------------------------------------------------------------
# Normalización de texto
# ---------------------------------------------------------------------------
def normalizar(texto: str) -> str:
    """Convierte a minúsculas y elimina acentos."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


def tokenizar(texto: str, min_len: int = 3) -> list[str]:
    """
    Tokeniza texto en palabras limpias, eliminando stopwords.
    Retorna lista de tokens normalizados.
    """
    if not texto:
        return []
    texto_norm = normalizar(texto)
    tokens = re.findall(r"[a-z]+", texto_norm)
    return [t for t in tokens if len(t) >= min_len and t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Lectores de warehouse
# ---------------------------------------------------------------------------
def leer_warehouse(base_path: str, años=None, meses=None, columnas=None):
    """
    Generador OPTIMIZADO que recorre el warehouse particionado.
    Lee columnas completas en batch (7-8x más rápido que fila a fila).
    Yields: dict con las columnas solicitadas de fact_news.

    Args:
        base_path: ruta a la carpeta fact_news/
        años:      lista de años a filtrar, ej. [2023, 2024]  (None = todos)
        meses:     lista de meses a filtrar, ej. [1, 2, 3]    (None = todos)
        columnas:  lista de columnas a leer, ej. ["year","month","title","body"]
                   (None = todas). Reducir columnas acelera mucho la lectura.
    """
    for entry in sorted(os.listdir(base_path)):
        if not entry.startswith("year="):
            continue
        anio = int(entry.split("=")[1])
        if años and anio not in años:
            continue
        year_path = os.path.join(base_path, entry)

        for sub in sorted(os.listdir(year_path)):
            if not sub.startswith("month="):
                continue
            mes = int(sub.split("=")[1])
            if meses and mes not in meses:
                continue
            month_path = os.path.join(year_path, sub)

            for fname in sorted(os.listdir(month_path)):
                if not fname.endswith(".parquet"):
                    continue
                fpath = os.path.join(month_path, fname)

                pf = pq.ParquetFile(fpath)
                table = pf.read(columns=columnas)

                # Castear columnas dictionary → int64 si hace falta
                for col_name in ["year", "month"]:
                    if col_name in table.schema.names:
                        field = table.schema.field(col_name)
                        if pa.types.is_dictionary(field.type):
                            idx = table.schema.get_field_index(col_name)
                            table = table.set_column(
                                idx, col_name,
                                table.column(col_name).cast(pa.int64())
                            )

                # Convertir cada columna a lista Python de una vez (mucho más rápido)
                cols_presentes = table.schema.names
                listas = {col: table.column(col).to_pylist() for col in cols_presentes}
                n = table.num_rows

                for i in range(n):
                    yield {col: listas[col][i] for col in cols_presentes}


def leer_parquet_simple(path: str) -> list[dict]:
    """Lee un archivo parquet pequeño (dimensiones) como lista de dicts."""
    table = pq.read_table(path)
    cols = table.schema.names
    filas = []
    for batch in table.to_batches():
        for i in range(batch.num_rows):
            filas.append({col: batch.column(col)[i].as_py() for col in cols})
    return filas


def cargar_dim_region(path: str) -> dict:
    """Retorna dict {region_sk: region_name}."""
    filas = leer_parquet_simple(path)
    return {f["region_sk"]: f["region_name"] for f in filas}


def cargar_dim_source(path: str) -> dict:
    """Retorna dict {source_sk: source}."""
    filas = leer_parquet_simple(path)
    return {f["source_sk"]: f["source"] for f in filas}


# ---------------------------------------------------------------------------
# Funciones MapReduce base
# ---------------------------------------------------------------------------

def shuffle_and_sort(pares):
    """
    Agrupa pares (clave, valor) por clave.
    Retorna lista de (clave, [valores]) ordenada por clave.
    """
    agrupado = {}
    for clave, valor in pares:
        if clave not in agrupado:
            agrupado[clave] = []
        agrupado[clave].append(valor)
    return sorted(agrupado.items())


def reduce_list(funcion, grupos):
    """
    Aplica funcion de reducción a cada grupo (clave, [valores]).
    Retorna generador de (clave, resultado_reducido).
    """
    for clave, valores in grupos:
        yield clave, functools.reduce(funcion, valores)


def sumar(a, b):
    """Función de reducción: suma dos valores."""
    return a + b


# ---------------------------------------------------------------------------
# Pipeline completo: map → shuffle → reduce
# ---------------------------------------------------------------------------
def mapreduce(mapper, reducer, iterable):
    """
    Ejecuta una pasada completa de MapReduce.

    Args:
        mapper: función que recibe un elemento y retorna (clave, valor)
        reducer: función binaria de reducción ej. lambda a, b: a + b
        iterable: fuente de datos

    Returns:
        lista de (clave, valor_reducido)
    """
    pares = list(map(mapper, iterable))
    grupos = shuffle_and_sort(pares)
    return list(reduce_list(reducer, grupos))


def top_k(resultados, k: int, reverse=True) -> list:
    """
    Retorna los k elementos con mayor valor de una lista de (clave, valor).
    """
    return sorted(resultados, key=lambda x: x[1], reverse=reverse)[:k]


# ---------------------------------------------------------------------------
# Helpers de conteo optimizados (equivalen a combiner en MapReduce real)
# ---------------------------------------------------------------------------

def contar_tokens_por_clave(iterable_articulos, clave_fn, texto_fn):
    """
    Versión optimizada del patrón map→shuffle→reduce para conteo de tokens.
    En lugar de emitir millones de pares (clave, 1), acumula directamente
    en un dict — equivale a aplicar un combiner local antes del shuffle.

    Args:
        iterable_articulos: generador de dicts de artículos
        clave_fn:  función articulo → clave (ej. año-mes, region_sk, source_sk)
        texto_fn:  función articulo → texto a tokenizar

    Returns:
        dict {(clave, token): count}
    """
    import collections
    contador = collections.defaultdict(int)
    for art in iterable_articulos:
        clave = clave_fn(art)
        if clave is None:
            continue
        for tok in tokenizar(texto_fn(art)):
            contador[(clave, tok)] += 1
    return dict(contador)