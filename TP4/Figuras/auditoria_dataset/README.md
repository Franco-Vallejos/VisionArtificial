# Auditoria de cuadrados y rectangulos ambiguos

Esta carpeta contiene copias recortadas de objetos cuya apariencia no coincide claramente con la etiqueta asignada. Los archivos originales de `train`, `valid` y `test` no fueron modificados.

## Carpetas

- `rectangulos_parecen_cuadrados`: 7 objetos etiquetados como `Rectangle` cuya proporcion orientada es menor que 1.25.
- `cuadrados_parecen_rectangulos`: 9 objetos etiquetados como `Square` cuya proporcion orientada es mayor que 1.5831.
- `resumen_casos_sospechosos.jpg`: montaje para comparar rápidamente los casos encontrados.

La proporcion orientada se calcula como `lado mayor / lado menor` sobre el contorno azul. Es una ayuda para localizar casos sospechosos, no una decision automatica definitiva: la perspectiva, las irregularidades del dibujo y la segmentacion del color pueden alterar la medicion.

El nombre de cada recorte conserva la particion, la etiqueta actual, la proporcion, la imagen de origen y el numero de objeto dentro de esa imagen.
