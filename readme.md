# Recomendación de películas

## Arquitectura

### Main.py
Encargado de realizar las llamadas de al chief reviewer, a la base de datos.
Será tanbien el que dado un usuario dara una recomendación de una nueva película.

### Chief-reviewer.py
Donde se ejecuta el feedback loop. Durante el entrenamiento recibe del main la terna
[Usuario(personalidad/definicion, peliculas_vistas[]), Pelicula(nombre, sipnosis), Rating_Usuario]-> genera Rating_predicho al consultar con los reviewers con personalidad y modifica pesos.
Para la fase de recomendación recibe del main la terna
[Usuario(personalidad/definicion, peliculas_vistas[]), Pelicula(nombre, sipnosis), Rating_Usuario] -> compara todas las peliculas y le dara la que más le gusta. (se podra comparar veracidad con cross-validation)

### reviewers.py
Es la clase que llama a OpenAI API, esta clase representa un conjunto de distintos reviewers donde cada uno tiene definida una personalidad en resources (que es el system_prompt), debemos de tener hay una llamada de api por cada uno de los usuario