import pickle
import gzip
import os

ruta_original = 'src/models/rf_precio.pkl'
ruta_comprimida = 'src/models/rf_precio.pkl.gz'

print("Cargando modelo original...")
with open(ruta_original, 'rb') as f:
    modelo = pickle.load(f)

print("Comprimiendo y guardando...")
with gzip.open(ruta_comprimida, 'wb') as f:
    pickle.dump(modelo, f)

tam_original = os.path.getsize(ruta_original) / (1024*1024)
tam_comprimido = os.path.getsize(ruta_comprimida) / (1024*1024)

print(f"¡Hecho! Tamaño original: {tam_original:.1f} MB -> Comprimido: {tam_comprimido:.1f} MB")