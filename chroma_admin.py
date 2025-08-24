# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 23:09:41 2025

@author: Admin
"""
import argparse, chromadb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="./chroma_db")
    ap.add_argument("--drop", nargs="*", default=[], help="Nombres de colecciones a borrar")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.path)

    if args.list:
        print("Colecciones existentes:")
        for name in client.list_collections():
            print(" -", name.name)
        return

    for col in args.drop:
        try:
            c = client.get_collection(col)
            client.delete_collection(col)
            print(f"✅ Borrada colección: {col}")
        except Exception as e:
            print(f"⚠️ No se pudo borrar {col}: {e}")

if __name__ == "__main__":
    main()
