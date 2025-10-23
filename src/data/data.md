Base de datos mínima para manejo de películas y calificaciones

Tablas:

- pelicula(id, nombre, sinopsis):
	- id: clave primaria autoincremental
	- nombre: texto, obligatorio
	- sinopsis: texto, opcional

- usuario(id, personalidad):
	- id: clave primaria autoincremental
	- personalidad: texto que describe al usuario

- peli_usuario(id, id_us, id_peli, cali_usuar):
	- id: clave primaria autoincremental
	- id_us: FK hacia `usuario(id)` (ON DELETE CASCADE)
	- id_peli: FK hacia `pelicula(id)` (ON DELETE CASCADE)
	- cali_usuar: float/numérico con rango (0–10)
	- UNIQUE(id_us, id_peli) para evitar duplicados por usuario/película

Notas breves:
- Usar CHECK para validar rango de `cali_usuar` (p. ej. >=0 y <=10).
- Se recomiendan índices sobre `peli_usuario(id_peli)` y `peli_usuario(id_us)` para consultas frecuentes.
- Si se quiere historial de calificaciones, añadir columna `fecha` y quitar la restricción UNIQUE.
