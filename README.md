#  Fluid Player ACG

Un reproductor de video experimental de código abierto con **interpolación de frames en tiempo real** usando RIFE.

##  Autor
**Arnold Cardenas** - Creador y Desarrollador Principal

##  ¿Cómo usarlo? (Guía de inicio rápido)

¡No necesitas instalar nada! Sigue estos pasos:

1. Ve a la sección de **Releases** (a la derecha de esta página).
2. Descarga el archivo `Fluid Player ACG.exe`.
3. Haz doble clic en el `.exe` para abrir el programa. *(La primera vez puede tardar unos segundos en abrir).*
4. Haz clic en el botón **"Abrir video"** en la esquina superior izquierda y selecciona tu video (.mp4, .mkv, .avi, etc.).
5. **¡Listo!** El video comenzará a reproducirse y la magia de la interpolación hará que se vea súper fluido.

###  Controles y Configuración
- **Interpolación (Casilla arriba):** Si la dejas marcada, el video se interpola (más fluido). Si la desmarcas, el video se reproduce normal (útil si tu PC no da abasto con un video muy pesado).
- **Menú "Reproducción":** 
  - *Calidad:* Elige entre "Calidad Plus" (mejor imagen, más pesado) o "Rendimiento" (menos resolución, más fluido). Si pones un video 4K, el programa bajará a "Rendimiento" automáticamente para no explotar.
  - *Motor:* Puedes elegir el motor RIFE estándar o los motores "Ligeros" (v4.25, v4.17, v4.15) si tu gráfica está sufriendo.
- **Menú "Audio" y "Subtítulo":** Si tu video tiene varias pistas de audio o subtítulos, aquí puedes cambiarlos.
- **Controles inferiores:** Play/Pausa, Retroceder 10s, Avanzar 10s, barra de progreso y volumen.
- **Pantalla completa:** Haz clic en el icono de la esquina inferior derecha (o presiona `Esc` para salir).

## Características
- **Interpolación en tiempo real:** Convierte videos de 24/30 FPS a 60 FPS fluidos.
- **Lector Universal:** Lee casi cualquier formato (.mp4, .mkv, .avi, etc.).
- **Hack del 4K:** Ajuste automático para que RIFE no colapse con videos pesados.
- **Sincronización Inteligente:** Sistema de "Reloj Maestro" para que el audio y el video no se separen.
- **Interfaz Moderna:** Diseño oscuro, menús redondeados y controles estilo Spotify.

## ¿Eres programador? (Instalación desde código)
Si quieres modificar el código, necesitarás Python 3.10+ y las librerías correspondientes (PyQt6, RIFE, PyAV, OpenCV, sounddevice). 

## Licencia 
este proyecto esta bajo la licencia MIT siente libre de usarlo modificarlo y compartirlo.