from symspellpy import SymSpell

from constants.constants_nlp import SYMSPELL_DICTIONARIES_PATHS

spell_es = SymSpell(
    max_dictionary_edit_distance=2, # Distancia de búsqueda
    prefix_length=7 # Prefijos de palabras
)
load = spell_es.load_dictionary(
    SYMSPELL_DICTIONARIES_PATHS["es"],
    term_index=0, # posicion donde se encuentran los terminos
    count_index=1, # posicion donde se encuentran las frecuencias
    encoding="utf-8"
)
if not load: print("[sympellpy]: No ha sido posible cargar el diccionario en español")

spell_en = SymSpell(
    max_dictionary_edit_distance=2, # Distancia de búsqueda
    prefix_length=7 # Prefijos de palabras
)
load = spell_en.load_dictionary(
    SYMSPELL_DICTIONARIES_PATHS["en"],
    term_index=0, # posicion donde se encuentran los terminos
    count_index=1, # posicion donde se encuentran las frecuencias
    encoding="utf-8"
)
if not load: print("[sympellpy]: No ha sido posible cargar el diccionario en inglés")

spellers = {
    "es": spell_es,
    "en": spell_en
}

def spell(texto, lang="es"):
    if lang not in SYMSPELL_DICTIONARIES_PATHS.keys():
        print(f"Idioma no soportado: {lang}")
        return None
    
    spell = spellers[lang]
    sugerencias = spell.lookup_compound(
        texto,
        max_edit_distance=2,
        ignore_non_words=True, # ignorar caracteres como números
    )
    return sugerencias[0].term