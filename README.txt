Simpsons S01 Multilingual Dataset (ES/EN/CA)
--------------------------------------------
Carpetas:
- simpsons_data_es: 13 episodios en español
- simpsons_data_en: 13 episodios en inglés
- simpsons_data_ca: 13 episodios en catalán

Formato de cada archivo:
  * ES: Título, Temporada, Episodio, Sinopsis, Personajes destacados
  * EN: Title, Season, Episode, Synopsis, Main characters
  * CA: Títol, Temporada, Episodi, Sinopsi, Personatges destacats

Sugerencias RAG + Chroma:
- Indexa cada idioma por separado en colecciones distintas para comparar recall/precision.
- Mantén los mismos filenames (s01eXX.txt) para facilitar análisis cruzado.
