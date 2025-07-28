# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 22:32:18 2025

@author: Admin
"""
from pathlib import Path

# Datos ficticios simplificados de la temporada 1 de Los Simpson
episodios = [
    {
        "titulo": "Simpsons Roasting on an Open Fire",
        "episodio": 1,
        "sinopsis": "En Navidad, Homer descubre que no recibirá bono navideño y toma un trabajo como Santa Claus. La familia adopta a Santa's Little Helper.",
        "personajes": "Homer, Bart, Marge, Santa's Little Helper"
    },
    {
        "titulo": "Bart the Genius",
        "episodio": 2,
        "sinopsis": "Bart hace trampa en un test de inteligencia y es enviado a una escuela para niños superdotados, donde no encaja.",
        "personajes": "Bart, Homer, Marge, Skinner"
    },
    {
        "titulo": "Homer's Odyssey",
        "episodio": 3,
        "sinopsis": "Homer es despedido y cae en depresión, pero luego se convierte en inspector de seguridad nuclear.",
        "personajes": "Homer, Marge, Bart, Mr. Burns"
    },
    {
        "titulo": "There's No Disgrace Like Home",
        "episodio": 4,
        "sinopsis": "Homer lleva a la familia a terapia para resolver sus problemas familiares.",
        "personajes": "Homer, Marge, Bart, Lisa"
    },
    {
        "titulo": "Bart the General",
        "episodio": 5,
        "sinopsis": "Bart se enfrenta al abusón Nelson con la ayuda del abuelo y Herman, formando un pequeño ejército escolar.",
        "personajes": "Bart, Nelson, Abe Simpson, Herman"
    },
    {
        "titulo": "Moaning Lisa",
        "episodio": 6,
        "sinopsis": "Lisa se siente deprimida y encuentra consuelo en la música y en el saxofonista Bleeding Gums Murphy.",
        "personajes": "Lisa, Bleeding Gums Murphy, Marge"
    },
    {
        "titulo": "The Call of the Simpsons",
        "episodio": 7,
        "sinopsis": "La familia se pierde en el bosque durante un viaje de camping. Homer es confundido con Pie Grande.",
        "personajes": "Homer, Bart, Marge, Lisa, Maggie"
    },
    {
        "titulo": "The Telltale Head",
        "episodio": 8,
        "sinopsis": "Bart corta la cabeza de la estatua de Jebediah Springfield para impresionar a los chicos populares, causando conmoción en la ciudad.",
        "personajes": "Bart, Homer, Lisa, Jebediah Springfield"
    },
    {
        "titulo": "Life on the Fast Lane",
        "episodio": 9,
        "sinopsis": "Marge considera tener un romance con un instructor de boliche después de sentirse ignorada por Homer.",
        "personajes": "Marge, Homer, Jacques"
    },
    {
        "titulo": "Homer's Night Out",
        "episodio": 10,
        "sinopsis": "Bart toma una foto comprometedora de Homer en una fiesta y la ciudad lo ve como un mujeriego, afectando su relación con Marge.",
        "personajes": "Homer, Marge, Bart"
    },
    {
        "titulo": "The Crepes of Wrath",
        "episodio": 11,
        "sinopsis": "Bart es enviado a Francia en un programa de intercambio estudiantil donde lo tratan como esclavo. Mientras, un niño albanés espía la planta nuclear.",
        "personajes": "Bart, Homer, Adil"
    },
    {
        "titulo": "Krusty Gets Busted",
        "episodio": 12,
        "sinopsis": "Krusty es arrestado por un robo que cometió su ayudante Bob. Bart y Lisa investigan para probar la inocencia de Krusty.",
        "personajes": "Bart, Lisa, Krusty, Sideshow Bob"
    },
    {
        "titulo": "Some Enchanted Evening",
        "episodio": 13,
        "sinopsis": "Homer y Marge salen por la noche y contratan a una niñera que resulta ser una ladrona buscada.",
        "personajes": "Homer, Marge, Bart, Lisa, Maggie"
    },
]

# Crear carpeta si no existe
data_dir = Path("simpsons_data")
data_dir.mkdir(exist_ok=True)

# Crear los archivos de texto
for ep in episodios:
    num = f"s01e{ep['episodio']:02d}"
    content = f"""Título: {ep['titulo']}
Temporada: 1
Episodio: {ep['episodio']}
Sinopsis: {ep['sinopsis']}
Personajes destacados: {ep['personajes']}
"""
    with open(data_dir / f"{num}.txt", "w", encoding="utf-8") as f:
        f.write(content)

# Mostrar la lista de archivos generados
import pandas as pd

df = pd.DataFrame([{
    "archivo": f"s01e{ep['episodio']:02d}.txt",
    "título": ep["titulo"],
    "personajes": ep["personajes"]
} for ep in episodios])
