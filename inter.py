"""
Fluid Player ACG (Open Source Edition)
======================================
Creador y Desarrollador Principal: Arnold Cardenas
Año: 2026

Descripción:
Reproductor de video experimental con interpolación de frames en tiempo real 
usando RIFE y decodificación híbrida (PyAV/OpenCV) para máxima compatibilidad.
"""

import sys
import os
import cv2
import time
import subprocess
import gc
import queue
import re
import json
import av
import sounddevice as sd
import numpy as np
from rife_ncnn_vulkan_python import Rife
from PIL import Image
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QLabel, QCheckBox, QProgressBar, QSlider, QMessageBox, QComboBox, QMenu, QMenuBar
from PyQt6.QtCore import QTimer, Qt, QEvent, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QTransform, QCursor, QPen, QPainter, QColor, QBrush, QPolygon
from PyQt6.QtCore import QPoint
import math

if getattr(sys, 'frozen', False):
    CARPETA_BASE = sys._MEIPASS
    CARPETA_DATOS = os.path.join(os.environ.get("APPDATA", os.path.dirname(sys.executable)), "FluidPlayer")
else:
    CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))
    CARPETA_DATOS = CARPETA_BASE

os.makedirs(CARPETA_DATOS, exist_ok=True)

FFMPEG = os.path.join(CARPETA_BASE, "ffmpeg", "ffmpeg.exe")
CARPETA_MODELO_RIFE = os.path.join(CARPETA_BASE, "model_r", "rife-v4.6")
ARCHIVO_CONFIGURACION = os.path.join(CARPETA_DATOS, "configuracion.json")


# ==========================================
# LECTOR UNIVERSAL
# ==========================================
class LectorUniversal:
    def __init__(self, ruta_video):
        self.ruta = ruta_video
        self.cap = None
        self.contenedor = None
        self.stream_video = None
        self.iterador_frames = None
        self.fps = 0.0
        self.total_frames = 0
        self.es_pyav = False
        self.abierto = False

        try:
            self.contenedor = av.open(ruta_video, options={'err_detect': 'ignore_err', 'analyzeduration': '100M', 'probesize': '100M'})
            self.stream_video = self.contenedor.streams.video[0]
            self.fps = float(self.stream_video.average_rate)
            self.total_frames = self.stream_video.frames
            if self.total_frames == 0:
                self.total_frames = int(self.stream_video.duration * self.fps / av.time_base)
            self.iterador_frames = self.contenedor.decode(self.stream_video)
            self.es_pyav = True
            self.abierto = True
            print("INFO: Video cargado con PyAV (Modo compatible).")
            return
        except Exception as e:
            print(f"PyAV no pudo abrirlo: {e}. Intentando con OpenCV...")
            if self.contenedor:
                self.contenedor.close()

        try:
            self.cap = cv2.VideoCapture(ruta_video)
            if self.cap.isOpened():
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.es_pyav = False
                self.abierto = True
                print("INFO: Video cargado con OpenCV (Modo rápido).")
            else:
                self.cap.release()
                self.cap = None
        except Exception as e:
            print(f"OpenCV también falló: {e}")

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        elif prop_id == cv2.CAP_PROP_POS_MSEC:
            if self.es_pyav and self.stream_video:
                return (self.stream_video.frames * 1000.0) / self.fps if self.stream_video.frames else 0.0
            elif self.cap:
                return self.cap.get(prop_id)
            return 0.0
        return 0.0

    def set(self, prop_id, valor):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            if self.es_pyav:
                tiempo = valor / self.fps
                self.contenedor.seek(int(tiempo * av.time_base), stream=self.stream_video)
                self.iterador_frames = self.contenedor.decode(self.stream_video)
            elif self.cap:
                self.cap.set(prop_id, valor)

    def read(self):
        if self.es_pyav:
            try:
                frame_av = next(self.iterador_frames)
                frame_ndarray = frame_av.to_ndarray(format='bgr24')
                return True, frame_ndarray
            except StopIteration:
                return False, None
            except Exception as e:
                print(f"Error en PyAV read: {e}")
                return False, None
        elif self.cap:
            return self.cap.read()
        return False, None

    def isOpened(self):
        return self.abierto

    def release(self):
        if self.cap:
            self.cap.release()
        if self.contenedor:
            self.contenedor.close()


class HiloCarga(QThread):
    progreso = pyqtSignal(str)
    terminado = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, ruta_video, carpeta_modelo_rife, factor_calidad=0.50):
        super().__init__()
        self.ruta_video = ruta_video
        self.carpeta_modelo_rife = carpeta_modelo_rife
        self.factor_calidad = factor_calidad

    def run(self):
        try:
            self.progreso.emit("Optimizando tu experiencia")

            pistas_audio = []
            pistas_subtitulos = []
            try:
                contenedor_av = av.open(self.ruta_video)
                for i, s in enumerate(contenedor_av.streams.audio):
                    idioma = s.language if s.language else f"Pista {i}"
                    pistas_audio.append((i, idioma))
                for i, s in enumerate(contenedor_av.streams.subtitles):
                    idioma = s.language if s.language else f"Pista {i}"
                    pistas_subtitulos.append((i, idioma))
                contenedor_av.close()
            except Exception:
                pistas_audio = []
                pistas_subtitulos = []

            cap = LectorUniversal(self.ruta_video)

            if not cap.isOpened():
                self.error.emit("No se pudo abrir este archivo. Puede que no sea un video válido o el códec no esté soportado.")
                return

            fps_original = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duracion_total_segundos = total_frames / fps_original if fps_original > 0 else 0

            ret, primer_frame = cap.read()
            if not ret:
                self.error.emit("No se pudo leer el video. El archivo puede estar dañado o el códec no está soportado.")
                return

            alto_original, ancho_original = primer_frame.shape[:2]

            try:
                contenedor_sar = av.open(self.ruta_video)
                stream_video = contenedor_sar.streams.video[0]
                sar = stream_video.sample_aspect_ratio
                if sar and float(sar) != 1.0:
                    ancho_original = int(ancho_original * float(sar))
                contenedor_sar.close()
            except Exception:
                pass

            interpolacion_disponible = True
            rife_elegido = None
            factor_elegido = self.factor_calidad

            # Arreglo de múltiplos de 8
            ancho_bruto = int(ancho_original * factor_elegido)
            alto_bruto = int(alto_original * factor_elegido)
            ancho_final = max(8, (ancho_bruto // 8) * 8)
            alto_final = max(8, (alto_bruto // 8) * 8)

            self.progreso.emit("Optimizando tu experiencia")
            try:
                rife_elegido = Rife(gpuid=0, model=str(self.carpeta_modelo_rife), width=ancho_final, height=alto_final)
            except Exception:
                interpolacion_disponible = False

            resultado = {
                "cap": cap,
                "fps_original": fps_original,
                "duracion_total_segundos": duracion_total_segundos,
                "primer_frame": primer_frame,
                "ancho_original": ancho_original,
                "alto_original": alto_original,
                "interpolacion_disponible": interpolacion_disponible,
                "rife": rife_elegido,
                "ancho_final": ancho_final,
                "alto_final": alto_final,
                "factor_elegido": factor_elegido,
                "pistas_audio": pistas_audio,
                "pistas_subtitulos": pistas_subtitulos,
            }
            self.terminado.emit(resultado)

        except Exception as error:
            self.error.emit(str(error))


class DecoderWorker(QThread):
    error = pyqtSignal(str)

    def __init__(self, cap, cola_salida, ancho_final, alto_final, segundo_inicio=0.0, fps_original=24.0):
        super().__init__()
        self.cap = cap
        self.cola_salida = cola_salida
        self.ancho_final = ancho_final
        self.alto_final = alto_final
        self._corriendo = True
        if segundo_inicio > 0:
            frame_inicio = int(segundo_inicio * fps_original)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

    def detener(self):
        self._corriendo = False

    def run(self):
        try:
            contador_debug = 0
            while self._corriendo:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"DECODER: video terminado tras {contador_debug} frames")
                    self.cola_salida.put(None)
                    break
                contador_debug += 1
                if contador_debug <= 3:
                    print(f"DECODER: frame {contador_debug} leído correctamente")

                frame_chico = cv2.resize(frame, (self.ancho_final, self.alto_final))
                frame_rgb = cv2.cvtColor(frame_chico, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)

                while self._corriendo:
                    try:
                        self.cola_salida.put((img_pil, frame_chico), timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception as error:
            self.error.emit(str(error))


class RifeWorker(QThread):
    error = pyqtSignal(str)

    def __init__(self, rife, cola_entrada, cola_salida, fps_original, fps_salida=60):
        super().__init__()
        self.rife = rife
        self.cola_entrada = cola_entrada
        self.cola_salida = cola_salida
        self.fps_original = fps_original
        self.fps_salida = fps_salida
        self._corriendo = True

    def detener(self):
        self._corriendo = False

    def run(self):
        try:
            frame_anterior = None
            ratio = self.fps_salida / self.fps_original
            acumulador = 0.0
            frames_emitidos_totales = 0
            deuda_tiempo = 0.0
            tiempo_por_frame_salida = 1 / self.fps_salida

            while self._corriendo:
                try:
                    item = self.cola_entrada.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is None:
                    self._meter_en_cola_salida(None)
                    break

                img_pil, frame_cv2 = item

                if frame_anterior is None:
                    frame_anterior = (img_pil, frame_cv2)
                    self._meter_en_cola_salida(frame_cv2)
                    frames_emitidos_totales += 1
                    continue

                img0_pil, frame0_cv2 = frame_anterior

                acumulador += ratio
                frames_a_generar = int(acumulador) - frames_emitidos_totales
                frames_a_generar = max(1, frames_a_generar)

                presupuesto = tiempo_por_frame_salida * frames_a_generar

                if deuda_tiempo > presupuesto * 3:
                    for _ in range(frames_a_generar):
                        self._meter_en_cola_salida(frame_cv2)
                    deuda_tiempo -= presupuesto
                else:
                    inicio = time.time()
                    for k in range(1, frames_a_generar):
                        timestep = k / frames_a_generar
                        try:
                            interp = self.rife.process(img0_pil, img_pil, timestep=timestep)
                            frame_interp_cv2 = np.array(interp)
                            frame_interp_cv2 = cv2.cvtColor(frame_interp_cv2, cv2.COLOR_RGB2BGR)
                            self._meter_en_cola_salida(frame_interp_cv2)
                        except Exception:
                            self._meter_en_cola_salida(frame_cv2)
                    self._meter_en_cola_salida(frame_cv2)
                    tiempo_real = time.time() - inicio
                    deuda_tiempo += (tiempo_real - presupuesto)
                    deuda_tiempo = max(0.0, deuda_tiempo)

                frames_emitidos_totales += frames_a_generar
                frame_anterior = (img_pil, frame_cv2)

        except Exception as error:
            self.error.emit(str(error))

    def _meter_en_cola_salida(self, item):
        while self._corriendo:
            try:
                self.cola_salida.put(item, timeout=0.5)
                return
            except queue.Full:
                continue


class AudioWorker(QThread):
    error = pyqtSignal(str)

    def __init__(self, ruta_video, segundo_inicio=0.0, indice_pista=0):
        super().__init__()
        self.ruta_video = ruta_video
        self.indice_pista = indice_pista
        self._corriendo = True
        self._pausado = False
        self.segundos_reproducidos = segundo_inicio
        self._segundo_inicio = segundo_inicio
        self._stream = None
        self.volumen = 1.0
        self._reloj_inicio = None
        self._tiempo_audio_inicio = segundo_inicio

    def establecer_volumen(self, valor):
        self.volumen = valor

    def detener(self):
        self._corriendo = False

    def pausar(self):
        self._pausado = True

    def reanudar(self):
        self._pausado = False
        self._reloj_inicio = time.perf_counter()
        self._tiempo_audio_inicio = self.segundos_reproducidos

    def obtener_posicion(self):
        if self._pausado or self._reloj_inicio is None:
            return self.segundos_reproducidos
            
        tiempo_transcurrido = time.perf_counter() - self._reloj_inicio
        posicion_actual = self._tiempo_audio_inicio + tiempo_transcurrido
        
        if posicion_actual > self.segundos_reproducidos + 0.5:
            return self.segundos_reproducidos
            
        return posicion_actual

    def run(self):
        try:
            contenedor = av.open(self.ruta_video)
            pista_audio = contenedor.streams.audio[self.indice_pista]
            frecuencia_muestreo = pista_audio.rate

            if self._segundo_inicio > 0:
                contenedor.seek(int(self._segundo_inicio * av.time_base), any_frame=False)

            self._stream = sd.OutputStream(
                samplerate=frecuencia_muestreo,
                channels=pista_audio.channels,
                dtype='float32'
            )
            self._stream.start()

            resampleador = av.audio.resampler.AudioResampler(
                format='fltp', layout=pista_audio.layout, rate=frecuencia_muestreo
            )

            ya_alcanzo_el_punto = (self._segundo_inicio == 0)

            for paquete in contenedor.demux(pista_audio):
                if not self._corriendo:
                    break

                while self._pausado and self._corriendo:
                    self.msleep(50)

                for frame_audio in paquete.decode():
                    tiempo_frame = float(frame_audio.pts * frame_audio.time_base)

                    if not ya_alcanzo_el_punto:
                        if tiempo_frame + (frame_audio.samples / frecuencia_muestreo) < self._segundo_inicio:
                            continue
                        ya_alcanzo_el_punto = True
                        self.segundos_reproducidos = tiempo_frame
                        self._tiempo_audio_inicio = tiempo_frame
                        
                    if self._reloj_inicio is None:
                        self._reloj_inicio = time.perf_counter()
                        self._tiempo_audio_inicio = self.segundos_reproducidos

                    frames_remuestreados = resampleador.resample(frame_audio)
                    for frame_remuestreado in frames_remuestreados:
                        datos = frame_remuestreado.to_ndarray()
                        datos = np.ascontiguousarray(datos.T.astype(np.float32))

                        if not self._corriendo:
                            break

                        self._stream.write(datos * self.volumen)
                        self.segundos_reproducidos += len(datos) / frecuencia_muestreo

            self._stream.stop()
            self._stream.close()

        except Exception as error:
            self.error.emit(str(error))

class SubtitleWorker(QThread):
    error = pyqtSignal(str)
    subtitulo_actual = pyqtSignal(str)

    def __init__(self, ruta_video, indice_pista, obtener_tiempo_actual):
        super().__init__()
        self.ruta_video = ruta_video
        self.indice_pista = indice_pista
        self.obtener_tiempo_actual = obtener_tiempo_actual
        self._corriendo = True

    def detener(self):
        self._corriendo = False

    def run(self):
        try:
            contenedor = av.open(self.ruta_video)
            pista_sub = contenedor.streams.subtitles[self.indice_pista]

            eventos = []

            for paquete in contenedor.demux(pista_sub):
                if not self._corriendo:
                    break
                if paquete.pts is None:
                    continue

                inicio = float(paquete.pts * pista_sub.time_base)
                duracion = float(paquete.duration * pista_sub.time_base) if paquete.duration else 3.0
                fin = inicio + duracion

                for subtitulo in paquete.decode():
                    texto = ""
                    if hasattr(subtitulo, 'dialogue') and subtitulo.dialogue:
                        texto = subtitulo.dialogue
                    elif hasattr(subtitulo, 'text') and subtitulo.text:
                        texto = subtitulo.text

                    if isinstance(texto, bytes):
                        texto = texto.decode('utf-8', errors='replace')

                    if texto:
                        texto = re.sub(r'\{[^}]*\}', '', texto)
                        texto = texto.replace('\\N', '\n').replace('\\n', '\n')

                    if texto and texto.strip():
                        eventos.append((inicio, fin, texto.strip()))

            contenedor.close()
            print(f"SubtitleWorker: {len(eventos)} eventos de subtítulo cargados")

            ultimo_mostrado = None

            while self._corriendo:
                tiempo_actual = self.obtener_tiempo_actual()
                texto_a_mostrar = ""
                for inicio, fin, texto in eventos:
                    if inicio <= tiempo_actual <= fin:
                        texto_a_mostrar = texto
                        break

                if texto_a_mostrar != ultimo_mostrado:
                    self.subtitulo_actual.emit(texto_a_mostrar)
                    ultimo_mostrado = texto_a_mostrar

                self.msleep(100)

        except Exception as error:
            self.error.emit(str(error))


class BotonIcono(QPushButton):
    def __init__(self, tipo, parent=None):
        super().__init__(parent)
        self.tipo = tipo
        self.efecto_click = 0
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 17px;
            }
            QPushButton:hover {
                background-color: rgba(90, 140, 255, 0.15);
            }
            QPushButton:pressed {
                background-color: rgba(90, 140, 255, 0.4);
            }
            QPushButton:disabled {
                background-color: transparent;
            }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        color = QColor("#ffffff") if self.isEnabled() else QColor("#55555a")
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)

        cx = self.width() // 2
        cy = self.height() // 2

        if self.tipo == "play":
            triangulo = QPolygon([QPoint(cx - 7, cy - 8), QPoint(cx - 7, cy + 8), QPoint(cx + 8, cy)])
            painter.drawPolygon(triangulo)
        elif self.tipo == "pausa":
            painter.drawRect(cx - 8, cy - 8, 5, 16)
            painter.drawRect(cx + 3, cy - 8, 5, 16)
        elif self.tipo == "retroceder":
            t1 = QPolygon([QPoint(cx - 1, cy - 8), QPoint(cx - 1, cy + 8), QPoint(cx - 8, cy)])
            t2 = QPolygon([QPoint(cx + 6, cy - 8), QPoint(cx + 6, cy + 8), QPoint(cx - 1, cy)])
            painter.drawPolygon(t1)
            painter.drawPolygon(t2)
        elif self.tipo == "avanzar":
            t1 = QPolygon([QPoint(cx - 6, cy - 8), QPoint(cx - 6, cy + 8), QPoint(cx + 1, cy)])
            t2 = QPolygon([QPoint(cx + 1, cy - 8), QPoint(cx + 1, cy + 8), QPoint(cx + 8, cy)])
            painter.drawPolygon(t1)
            painter.drawPolygon(t2)
        elif self.tipo == "expandir":
            pluma = QPen(color, 2)
            pluma.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pluma)
            painter.drawLine(cx - 8, cy - 8, cx - 8, cy - 3)
            painter.drawLine(cx - 8, cy - 8, cx - 3, cy - 8)
            painter.drawLine(cx + 8, cy - 8, cx + 8, cy - 3)
            painter.drawLine(cx + 8, cy - 8, cx + 3, cy - 8)
            painter.drawLine(cx - 8, cy + 8, cx - 8, cy + 3)
            painter.drawLine(cx - 8, cy + 8, cx - 3, cy + 8)
            painter.drawLine(cx + 8, cy + 8, cx + 8, cy + 3)
            painter.drawLine(cx + 8, cy + 8, cx + 3, cy + 8)
        elif self.tipo == "contraer":
            pluma = QPen(color, 2)
            pluma.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pluma)
            painter.drawLine(cx - 8, cy - 3, cx - 8, cy - 8)
            painter.drawLine(cx - 8, cy - 8, cx - 3, cy - 8)
            painter.drawLine(cx + 8, cy - 3, cx + 8, cy - 8)
            painter.drawLine(cx + 8, cy - 8, cx + 3, cy - 8)
            painter.drawLine(cx - 8, cy + 3, cx - 8, cy + 8)
            painter.drawLine(cx - 8, cy + 8, cx - 3, cy + 8)
            painter.drawLine(cx + 8, cy + 3, cx + 8, cy + 8)
            painter.drawLine(cx + 8, cy + 8, cx + 3, cy + 8)
        elif self.tipo == "engranaje":
            painter.drawEllipse(cx - 4, cy - 4, 8, 8)
            pluma = QPen(color, 2)
            pluma.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pluma)
            for angulo in range(0, 360, 45):
                rad = math.radians(angulo)
                x1 = cx + 5 * math.cos(rad)
                y1 = cy + 5 * math.sin(rad)
                x2 = cx + 9 * math.cos(rad)
                y2 = cy + 9 * math.sin(rad)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.end()


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fluid Player ACG - Open Source Edition")
        ruta_logo = os.path.join(CARPETA_BASE, "logo.jpeg")
        ruta_logo_principal = os.path.join(CARPETA_BASE, "logoP.jpeg")
        if os.path.exists(ruta_logo_principal):
            self.setWindowIcon(QIcon(ruta_logo_principal))
        elif os.path.exists(ruta_logo):
            self.setWindowIcon(QIcon(ruta_logo))

        self.setGeometry(100, 100, 1000, 750)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #1a1a1a;")

        barra_menu = self.menuBar()
        barra_menu.setStyleSheet("""
            QMenuBar { background-color: #1e1e26; color: #f0f0f5; font-family: 'Segoe UI', sans-serif; font-size: 12px; padding: 2px; }
            QMenuBar::item { padding: 6px 12px; background-color: transparent; border-radius: 6px; }
            QMenuBar::item:selected { background-color: #3d3d4a; }
            QMenu { background-color: #2b2b35; color: #f0f0f5; border: 1px solid #3a3a45; border-radius: 8px; padding: 5px; }
            QMenu::item { padding: 6px 20px; border-radius: 6px; }
            QMenu::item:selected { background-color: #5a8cff; }
        """)

        self.menu_reproduccion = barra_menu.addMenu("Reproducción")
        self.menu_audio = barra_menu.addMenu("Audio")
        self.menu_subtitulos = barra_menu.addMenu("Subtítulo")

        estilo_boton = """
            QPushButton {
                background-color: #2b2b35; color: #f0f0f5; padding: 8px 14px;
                border-radius: 10px; border: 1px solid #3a3a45;
                font-size: 12px; font-family: 'Segoe UI', sans-serif; font-weight: 500;
            }
            QPushButton:hover { background-color: #3d3d4a; border: 1px solid #5a8cff; }
            QPushButton:pressed { background-color: #2a2a35; }
            QPushButton:disabled { background-color: #1a1a1e; color: #55555a; border: 1px solid #2a2a30; }
        """

        self.boton_abrir = QPushButton("Abrir video")
        self.boton_abrir.clicked.connect(self.abrir_video)
        self.boton_abrir.setStyleSheet(estilo_boton)
        self.boton_abrir.setMaximumWidth(100)

        self.checkbox_interpolacion = QCheckBox("Interpolación")
        self.checkbox_interpolacion.setChecked(True)
        self.checkbox_interpolacion.setStyleSheet("color: white; font-size: 12px;")

        self.label_fps = QLabel("")
        self.label_fps.setStyleSheet("color: #888; font-size: 11px;")
        self.label_fps.hide()

        barra_superior = QHBoxLayout()
        barra_superior.addWidget(self.boton_abrir)
        barra_superior.addWidget(self.checkbox_interpolacion)
        barra_superior.addStretch()
        barra_superior.addWidget(self.label_fps)
        barra_superior.setContentsMargins(0, 0, 0, 0)

        self.label_video = QLabel("Selecciona un video para reproducir")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_video.setStyleSheet("background-color: black; color: white; border: none; margin: 0px; padding: 0px;")
        self.label_video.setMouseTracking(True)

        self.label_mensaje_carga = QLabel("", self.label_video)
        self.label_mensaje_carga.setStyleSheet("""
            color: #cccccc; font-size: 13px; font-family: 'Segoe UI', 'Century Gothic', sans-serif;
            font-weight: 300; letter-spacing: 0.5px; background-color: transparent;
        """)
        self.label_mensaje_carga.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_mensaje_carga.hide()

        self.label_subtitulo = QLabel("", self.label_video)
        self.label_subtitulo.setStyleSheet("""
            color: white; font-size: 20px; font-family: 'Segoe UI', sans-serif;
            font-weight: 600; background-color: rgba(0, 0, 0, 140); padding: 6px; border-radius: 6px;
        """)
        self.label_subtitulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_subtitulo.setWordWrap(True)
        self.label_subtitulo.hide()

        self.pixmap_logo = QPixmap(ruta_logo) if os.path.exists(ruta_logo) else None
        self.angulo_rotacion_logo = 0
        self.timer_rotacion_logo = QTimer()
        self.timer_rotacion_logo.timeout.connect(self._rotar_logo)

        self.boton_retroceder = BotonIcono("retroceder")
        self.boton_retroceder.clicked.connect(lambda: self.saltar_relativo(-10))
        self.boton_retroceder.setEnabled(False)
        self.boton_retroceder.setFixedSize(34, 34)

        self.boton_play_pausa = BotonIcono("pausa")
        self.boton_play_pausa.clicked.connect(self.alternar_play_pausa)
        self.boton_play_pausa.setEnabled(False)
        self.boton_play_pausa.setFixedSize(34, 34)

        self.boton_avanzar = BotonIcono("avanzar")
        self.boton_avanzar.clicked.connect(lambda: self.saltar_relativo(10))
        self.boton_avanzar.setEnabled(False)
        self.boton_avanzar.setFixedSize(34, 34)

        self.barra_progreso = QProgressBar()
        self.barra_progreso.setRange(0, 1000)
        self.barra_progreso.setValue(0)
        self.barra_progreso.setTextVisible(False)
        self.barra_progreso.setMaximumHeight(10)
        self.barra_progreso.setMinimumHeight(10)
        self.barra_progreso.setStyleSheet("""
            QProgressBar {
                background-color: #2b2b35;
                border: 1px solid #3a3a45;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #5a8cff;
                border-radius: 5px;
            }
            QProgressBar::chunk:hover {
                background-color: #7ba3ff;
            }
        """)
        self.barra_progreso.mousePressEvent = self._clic_en_barra_progreso

        self.label_tiempo = QLabel("00:00 / 00:00")
        self.label_tiempo.setStyleSheet("color: #a0a0a0; font-size: 12px; font-family: 'Segoe UI', sans-serif; font-weight: 500;")
        self.label_tiempo.setMaximumWidth(120)

        self.label_volumen = QLabel("🔊")
        self.label_volumen.setStyleSheet("color: white;")
        self.slider_volumen = QSlider(Qt.Orientation.Horizontal)
        self.slider_volumen.setRange(0, 100)
        self.slider_volumen.setValue(100)
        self.slider_volumen.setMaximumWidth(80)
        self.slider_volumen.valueChanged.connect(self.cambiar_volumen)

        self.slider_volumen.setStyleSheet("""
            QSlider::groove:horizontal { background: #3a3a45; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #5a8cff; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
            QSlider::handle:horizontal:hover { background: #7ba3ff; }
            QSlider::sub-page:horizontal { background: #5a8cff; border-radius: 2px; }
        """)

        self.combo_calidad = QComboBox(self)
        self.combo_calidad.addItem("Calidad Plus", 0.75)
        self.combo_calidad.addItem("Calidad", 0.66)
        self.combo_calidad.addItem("Equilibrado", 0.50)
        self.combo_calidad.addItem("Rendimiento", 0.35)
        self.combo_calidad.setEnabled(False)
        self.combo_calidad.currentIndexChanged.connect(self.cambiar_calidad_manual)
        self.combo_calidad.hide()

        self.combo_motor = QComboBox(self)
        self.combo_motor.addItem("Motor Estandar", "rife-v4.6")
        self.combo_motor.addItem("Motor Ligero v4.25", "rife-v4.25-lite")
        self.combo_motor.addItem("Motor Ligero v4.17", "rife-v4.17-lite")
        self.combo_motor.addItem("Motor Ligero v4.15", "rife-v4.15-lite")
        self.combo_motor.setEnabled(False)
        self.combo_motor.currentIndexChanged.connect(self.cambiar_calidad_manual)
        self.combo_motor.hide()

        self.combo_audio = QComboBox(self)
        self.combo_audio.setEnabled(False)
        self.combo_audio.currentIndexChanged.connect(self.cambiar_pista_audio)
        self.combo_audio.hide()

        self.combo_subtitulos = QComboBox(self)
        self.combo_subtitulos.addItem("Subtítulos: Desactivados", None)
        self.combo_subtitulos.setEnabled(False)
        self.combo_subtitulos.currentIndexChanged.connect(self.cambiar_pista_subtitulos)
        self.combo_subtitulos.hide()

        self.boton_pantalla_completa = BotonIcono("expandir")
        self.boton_pantalla_completa.clicked.connect(self.alternar_pantalla_completa)
        self.boton_pantalla_completa.setFixedSize(34, 34)

        barra_inferior = QHBoxLayout()
        barra_inferior.addWidget(self.boton_retroceder)
        barra_inferior.addWidget(self.boton_play_pausa)
        barra_inferior.addWidget(self.boton_avanzar)
        barra_inferior.addWidget(self.barra_progreso)
        barra_inferior.addWidget(self.label_tiempo)
        barra_inferior.addWidget(self.label_volumen)
        barra_inferior.addWidget(self.slider_volumen)
        barra_inferior.addWidget(self.boton_pantalla_completa)
        
        barra_inferior.setContentsMargins(10, 5, 10, 5)
        barra_inferior.setSpacing(10)

        layout = QVBoxLayout()
        layout.addLayout(barra_superior)
        layout.addWidget(self.label_video, stretch=1)
        layout.addLayout(barra_inferior)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        contenedor = QWidget()
        contenedor.setLayout(layout)
        self.setCentralWidget(contenedor)

        self.cap = None
        self.rife = None
        self.timer = QTimer()
        self.timer_ocultar_controles = QTimer()
        self.timer_ocultar_controles.setSingleShot(True)
        self.timer_ocultar_controles.timeout.connect(self.ocultar_controles)

        self.posicion_mouse_anterior = None
        self.timer_vigilar_mouse = QTimer()
        self.timer_vigilar_mouse.timeout.connect(self._vigilar_posicion_mouse)
        self.timer_vigilar_mouse.start(300)

        self.timer.timeout.connect(self.mostrar_siguiente_frame)

        self.mostrando_interpolado = False
        self.img0_pil = None
        self.frame_actual_chico = None
        self.ancho_final = None
        self.alto_final = None
        self.numero_frame_mostrado = 0
        self.fps_original = None
        self.tiempo_por_frame_interpolado = None
        self.saltar_interpolacion_temporalmente = False
        self.pausado = False
        self.contador_fps = 0
        self.tiempo_ultimo_conteo_fps = time.time()

        self.subtitle_worker = None
        self.indice_pista_subtitulos_actual = None

        self._activar_seguimiento_mouse_recursivo(self)

        self.combo_calidad.setEnabled(True)
        self.combo_motor.setEnabled(True)

        configuracion = self._cargar_configuracion()
        if "indice_calidad" in configuracion:
            self.combo_calidad.blockSignals(True)
            self.combo_calidad.setCurrentIndex(configuracion["indice_calidad"])
            self.combo_calidad.blockSignals(False)
        if "indice_motor" in configuracion:
            self.combo_motor.blockSignals(True)
            self.combo_motor.setCurrentIndex(configuracion["indice_motor"])
            self.combo_motor.blockSignals(False)

        self._reconstruir_menus()

    def _rotar_logo(self):
        if self.pixmap_logo is None:
            return
        self.angulo_rotacion_logo = (self.angulo_rotacion_logo + 10) % 360
        
        tiempo_actual = time.time()
        tamano_variable = int(150 + (math.sin(tiempo_actual * 3) * 20)) 
        
        transformacion = QTransform().rotate(self.angulo_rotacion_logo)
        logo_rotado = self.pixmap_logo.transformed(transformacion, Qt.TransformationMode.SmoothTransformation)
        
        logo_escalado = logo_rotado.scaled(tamano_variable, tamano_variable, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        self.label_video.setPixmap(logo_escalado)
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ancho_label = self.label_video.width()
        alto_label = self.label_video.height()
        self.label_mensaje_carga.setGeometry(0, alto_label // 2 + 90, ancho_label, 40)
        self.label_mensaje_carga.setText("Estamos optimizando tu experiencia para este video...")
        self.label_mensaje_carga.show()

    def _cargar_configuracion(self):
        if os.path.exists(ARCHIVO_CONFIGURACION):
            try:
                with open(ARCHIVO_CONFIGURACION, "r") as archivo:
                    return json.load(archivo)
            except Exception:
                return {}
        return {}

    def _guardar_configuracion(self):
        configuracion = {
            "indice_calidad": self.combo_calidad.currentIndex(),
            "indice_motor": self.combo_motor.currentIndex(),
        }
        try:
            with open(ARCHIVO_CONFIGURACION, "w") as archivo:
                json.dump(configuracion, archivo)
        except Exception:
            pass

    def abrir_video(self):
        ruta_video, _ = QFileDialog.getOpenFileName(
            self, "Selecciona un video", "", "Videos (*.mp4 *.mkv *.avi *.mov)"
        )
        if not ruta_video:
            return
        self.ruta_video_actual = ruta_video
        self.timer.stop()
        if hasattr(self, 'audio_worker') and self.audio_worker is not None:
            self.audio_worker.detener()
            self.audio_worker.wait(2000)

        if hasattr(self, 'decoder_worker') and self.decoder_worker is not None:
            self.decoder_worker.detener()
            self.decoder_worker.wait(2000)
        if hasattr(self, 'rife_worker') and self.rife_worker is not None:
            self.rife_worker.detener()
            self.rife_worker.wait(2000)
        if self.subtitle_worker is not None:
            self.subtitle_worker.detener()
            self.subtitle_worker.wait(2000)
            self.subtitle_worker = None
        self.label_subtitulo.hide()

        gc.collect()

        self.boton_abrir.setEnabled(False)
        self.boton_play_pausa.setEnabled(False)

        if self.pixmap_logo is not None:
            self.timer_rotacion_logo.start(50)
        else:
            self.timer_animacion_carga = QTimer()
            self.timer_animacion_carga.timeout.connect(lambda: self._actualizar_carga("Optimizando tu experiencia"))
            self.timer_animacion_carga.start(400)

        nombre_motor_guardado = self.combo_motor.currentData()
        carpeta_motor_guardado = os.path.join(CARPETA_BASE, "model_r", nombre_motor_guardado)
        self.hilo_carga = HiloCarga(ruta_video, carpeta_motor_guardado, factor_calidad=self.combo_calidad.currentData())
        self.hilo_carga.terminado.connect(self.timer_rotacion_logo.stop)
        self.hilo_carga.terminado.connect(self._carga_terminada)
        self.hilo_carga.error.connect(self.timer_rotacion_logo.stop)
        self.hilo_carga.error.connect(self._carga_con_error)
        self.hilo_carga.start()

    def _carga_con_error(self, mensaje):
        self.timer_rotacion_logo.stop()
        self.label_mensaje_carga.hide()
        if hasattr(self, 'timer_animacion_carga'):
            self.timer_animacion_carga.stop()

        QMessageBox.warning(self, "Error", mensaje)
        self.label_video.setText("Selecciona un video para reproducir")
        self.boton_abrir.setEnabled(True)

    def _carga_terminada(self, resultado):
        self.timer_rotacion_logo.stop()
        self.label_mensaje_carga.hide()
        if hasattr(self, 'timer_animacion_carga'):
            self.timer_animacion_carga.stop()

        self.cap = resultado["cap"]
        self.fps_original = resultado["fps_original"]
        self.duracion_total_segundos = resultado["duracion_total_segundos"]
        self.ancho_final = resultado["ancho_final"]
        self.alto_final = resultado["alto_final"]
        self.interpolacion_disponible = resultado["interpolacion_disponible"]
        
        self.ancho_original_corregido = resultado["ancho_original"]
        self.alto_original_corregido = resultado["alto_original"]

        if self.ancho_original_corregido >= 2560:
            print("¡Video pesado detectado (2K/4K)! Forzando perfil de Rendimiento.")
            self.combo_calidad.setCurrentIndex(3)
            
            factor_auto = self.combo_calidad.currentData()
            ancho_bruto = int(self.ancho_original_corregido * factor_auto)
            alto_bruto = int(self.alto_original_corregido * factor_auto)
            self.ancho_final = max(8, (ancho_bruto // 8) * 8)
            self.alto_final = max(8, (alto_bruto // 8) * 8)
            
            if self.interpolacion_disponible and resultado["rife"] is not None:
                nombre_motor = self.combo_motor.currentData()
                carpeta_motor = os.path.join(CARPETA_BASE, "model_r", nombre_motor)
                try:
                    self.rife = Rife(gpuid=0, model=carpeta_motor, width=self.ancho_final, height=self.alto_final)
                    resultado["rife"] = self.rife
                except Exception as e:
                    print(f"No se pudo recargar RIFE para 4K: {e}")

        if self.interpolacion_disponible:
            self.ancho_final = resultado["ancho_final"] if self.ancho_original_corregido < 2560 else self.ancho_final
            self.alto_final = resultado["alto_final"] if self.ancho_original_corregido < 2560 else self.alto_final
        else:
            self.ancho_final = resultado["ancho_original"]
            self.alto_final = resultado["alto_original"]

        print(f"RESOLUCION: original_corregido={self.ancho_original_corregido}x{self.alto_original_corregido}, ancho_final(RIFE)={self.ancho_final}x{self.alto_final}")

        self.rife = resultado["rife"]

        self.combo_audio.blockSignals(True)
        self.combo_audio.clear()
        for indice, idioma in resultado["pistas_audio"]:
            self.combo_audio.addItem(f"Audio: {idioma}", indice)
        self.combo_audio.setEnabled(len(resultado["pistas_audio"]) > 1)
        self.combo_audio.blockSignals(False)

        self.combo_subtitulos.blockSignals(True)
        self.combo_subtitulos.clear()
        self.combo_subtitulos.addItem("Subtítulos: Desactivados", None)
        for indice, idioma in resultado["pistas_subtitulos"]:
            self.combo_subtitulos.addItem(f"Subtítulos: {idioma} ({indice})", indice)
        self.combo_subtitulos.setEnabled(len(resultado["pistas_subtitulos"]) > 0)
        self.combo_subtitulos.blockSignals(False)
        
        self.indice_pista_audio_actual = self.combo_audio.currentData() if self.combo_audio.count() > 0 else 0
        self.indice_pista_subtitulos_actual = None

        if not self.interpolacion_disponible:
            QMessageBox.information(
                self, "Interpolación no disponible",
                "Tu PC no tiene suficiente rendimiento para esta función con este video.\nSe reproducirá normalmente, sin interpolación."
            )
            self.checkbox_interpolacion.setChecked(False)
            self.checkbox_interpolacion.setEnabled(False)
        else:
            self.checkbox_interpolacion.setEnabled(True)

        self.img0_pil, self.frame_actual_chico = self.preparar_frame(resultado["primer_frame"])

        self.audio_worker = AudioWorker(self.ruta_video_actual, indice_pista=self.indice_pista_audio_actual)
        self.audio_worker.error.connect(lambda msg: print(f"ERROR AudioWorker: {msg}"))
        self.audio_worker.start()

        self.numero_frame_mostrado = 0
        self.mostrando_interpolado = False
        self.saltar_interpolacion_temporalmente = False
        self.tiempo_por_frame_interpolado = 1 / 60

        self.boton_abrir.setEnabled(True)
        self.pausado = False
        self.boton_play_pausa.setEnabled(True)
        self.boton_retroceder.setEnabled(True)
        self.boton_avanzar.setEnabled(True)

        self.cola_frames_originales = queue.Queue(maxsize=15)

        self.decoder_worker = DecoderWorker(self.cap, self.cola_frames_originales, self.ancho_final, self.alto_final)
        self.decoder_worker.error.connect(lambda msg: print(f"ERROR DecoderWorker: {msg}"))
        self.decoder_worker.start()

        self.cola_frames_listos = queue.Queue(maxsize=15)
        if self.interpolacion_disponible and self.checkbox_interpolacion.isChecked():
            self.rife_worker = RifeWorker(self.rife, self.cola_frames_originales, self.cola_frames_listos, self.fps_original, fps_salida=60)
            self.rife_worker.error.connect(lambda msg: print(f"ERROR RifeWorker: {msg}"))
            self.rife_worker.start()
        else:
            self.rife_worker = None

        self._reconstruir_menus()    
        self.timer.start(1)

    def _actualizar_carga(self, mensaje_base):
        self._contador_animacion_carga = getattr(self, "_contador_animacion_carga", 0) + 1
        puntos = "." * (self._contador_animacion_carga % 4)
        self.label_video.setText(f"{mensaje_base}{puntos}")
        QApplication.processEvents()

    def _reconstruir_menus(self):
        self.menu_reproduccion.clear()
        self.submenu_calidad = QMenu("Calidad", self.menu_reproduccion)
        self.menu_reproduccion.addMenu(self.submenu_calidad)
        submenu_calidad = self.submenu_calidad
        for i in range(self.combo_calidad.count()):
            texto = self.combo_calidad.itemText(i)
            accion = submenu_calidad.addAction(texto)
            accion.setCheckable(True)
            accion.setChecked(i == self.combo_calidad.currentIndex())
            accion.triggered.connect(lambda checked, indice=i: self.combo_calidad.setCurrentIndex(indice))
        submenu_calidad.setEnabled(self.combo_calidad.isEnabled())

        self.submenu_motor = QMenu("Motor", self.menu_reproduccion)
        self.menu_reproduccion.addMenu(self.submenu_motor)
        submenu_motor = self.submenu_motor
        for i in range(self.combo_motor.count()):
            texto = self.combo_motor.itemText(i)
            accion = submenu_motor.addAction(texto)
            accion.setCheckable(True)
            accion.setChecked(i == self.combo_motor.currentIndex())
            accion.triggered.connect(lambda checked, indice=i: self.combo_motor.setCurrentIndex(indice))
        submenu_motor.setEnabled(self.combo_motor.isEnabled())

        self.menu_audio.clear()
        for i in range(self.combo_audio.count()):
            texto = self.combo_audio.itemText(i)
            accion = self.menu_audio.addAction(texto)
            accion.setCheckable(True)
            accion.setChecked(i == self.combo_audio.currentIndex())
            accion.triggered.connect(lambda checked, indice=i: self.combo_audio.setCurrentIndex(indice))
        self.menu_audio.setEnabled(self.combo_audio.isEnabled())

        self.menu_subtitulos.clear()
        for i in range(self.combo_subtitulos.count()):
            texto = self.combo_subtitulos.itemText(i)
            accion = self.menu_subtitulos.addAction(texto)
            accion.setCheckable(True)
            accion.setChecked(i == self.combo_subtitulos.currentIndex())
            accion.triggered.connect(lambda checked, indice=i: self.combo_subtitulos.setCurrentIndex(indice))
        self.menu_subtitulos.setEnabled(self.combo_subtitulos.isEnabled())    

    def preparar_frame(self, frame_cv2):
        frame_chico = cv2.resize(frame_cv2, (self.ancho_final, self.alto_final))
        frame_rgb = cv2.cvtColor(frame_chico, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb), frame_chico

    def pil_a_cv2(self, img_pil):
        frame_rgb = np.array(img_pil)
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def mostrar_frame_en_pantalla(self, frame_cv2):
        frame_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
        alto, ancho, canales = frame_rgb.shape
        bytes_por_linea = canales * ancho
        qimg = QImage(frame_rgb.data, ancho, alto, bytes_por_linea, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        pixmap_escalado = pixmap.scaled(
            self.label_video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.label_video.setPixmap(pixmap_escalado)

        ancho_label = self.label_video.width()
        alto_label = self.label_video.height()
        self.label_subtitulo.setGeometry(20, alto_label - 90, ancho_label - 40, 70)

        self.contador_fps += 1
        ahora = time.time()
        if ahora - self.tiempo_ultimo_conteo_fps >= 1.0:
            self.label_fps.setText(f"{self.contador_fps} FPS")
            self.contador_fps = 0
            self.tiempo_ultimo_conteo_fps = ahora

    def alternar_play_pausa(self):
        self.boton_play_pausa.efecto_click = 5
        self.boton_play_pausa.update()
        if self.pausado:
            self.audio_worker.reanudar()
            self.pausado = False
            self.boton_play_pausa.tipo = "pausa"
            self.boton_play_pausa.update()
            self.timer.start(1)
        else:
            self.audio_worker.pausar()
            self.pausado = True
            self.boton_play_pausa.tipo = "play"
            self.boton_play_pausa.update()
            self.timer.stop()

    def alternar_pantalla_completa(self, checked=False):
        if self.isFullScreen():
            self.showNormal()
            self.boton_pantalla_completa.tipo = "expandir"
            self.boton_pantalla_completa.update()
        else:
            self.showFullScreen()
            self.boton_pantalla_completa.tipo = "contraer"
            self.boton_pantalla_completa.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            self.boton_pantalla_completa.tipo = "expandir"
            self.boton_pantalla_completa.update()
            self.mostrar_controles()
        super().keyPressEvent(event)

    def _vigilar_posicion_mouse(self):
        posicion_actual = QCursor.pos()
        if posicion_actual != self.posicion_mouse_anterior:
            self.posicion_mouse_anterior = posicion_actual
            tiempo_ultimo_movimiento_mouse = time.time()
            self.mostrar_controles()
        elif self.isFullScreen() and hasattr(self, 'tiempo_ultimo_movimiento_mouse'):
            if time.time() - self.tiempo_ultimo_movimiento_mouse >= 3.0:
                self.ocultar_controles()

    def mostrar_controles(self):
        self.boton_abrir.setVisible(True)
        self.checkbox_interpolacion.setVisible(True)
        self.label_tiempo.setVisible(True)
        self.menuBar().setVisible(True)
        self.label_fps.setVisible(False)
        self.boton_play_pausa.setVisible(True)
        self.boton_retroceder.setVisible(True)
        self.boton_avanzar.setVisible(True)
        self.barra_progreso.setVisible(True)
        self.label_volumen.setVisible(True)
        self.slider_volumen.setVisible(True)
        self.boton_pantalla_completa.setVisible(True)
        self.timer_ocultar_controles.start(3000)

    def ocultar_controles(self):
        if self.isFullScreen():
            self.boton_abrir.setVisible(False)
            self.label_tiempo.setVisible(False)
            self.menuBar().setVisible(False)
            self.checkbox_interpolacion.setVisible(False)
            self.label_fps.setVisible(False)
            self.boton_play_pausa.setVisible(False)
            self.boton_retroceder.setVisible(False)
            self.boton_avanzar.setVisible(False)
            self.barra_progreso.setVisible(False)
            self.label_volumen.setVisible(False)
            self.slider_volumen.setVisible(False)
            self.boton_pantalla_completa.setVisible(False)

    def _activar_seguimiento_mouse_recursivo(self, widget):
        widget.setMouseTracking(True)
        for hijo in widget.findChildren(QWidget):
            hijo.setMouseTracking(True)

    def cambiar_volumen(self, valor):
        if hasattr(self, 'audio_worker') and self.audio_worker is not None:
            self.audio_worker.establecer_volumen(valor / 100)
        self.label_volumen.setText(f"Volumen: {valor}%")

    def cambiar_pista_audio(self):
        if self.cap is None:
            return
        nuevo_indice = self.combo_audio.currentData()
        if nuevo_indice is None:
            return
        self.indice_pista_audio_actual = nuevo_indice
        tiempo_actual = self.audio_worker.obtener_posicion() if hasattr(self, 'audio_worker') else 0.0
        self.saltar_a_posicion(tiempo_actual)

    def cambiar_pista_subtitulos(self):
        indice_elegido = self.combo_subtitulos.currentData()
        self.indice_pista_subtitulos_actual = indice_elegido

        if self.subtitle_worker is not None:
            self.subtitle_worker.detener()
            self.subtitle_worker.wait(2000)
            self.subtitle_worker = None

        self.label_subtitulo.hide()

        if indice_elegido is None or self.cap is None:
            return

        self.subtitle_worker = SubtitleWorker(
            self.ruta_video_actual, indice_elegido,
            lambda: self.audio_worker.obtener_posicion() if hasattr(self, 'audio_worker') else 0.0
        )
        self.subtitle_worker.subtitulo_actual.connect(self._actualizar_texto_subtitulo)
        self.subtitle_worker.error.connect(lambda msg: print(f"ERROR SubtitleWorker: {msg}"))
        self.subtitle_worker.start()

    def _actualizar_texto_subtitulo(self, texto):
        if texto:
            self.label_subtitulo.setText(texto)
            self.label_subtitulo.show()
        else:
            self.label_subtitulo.hide()   

    def cambiar_calidad_manual(self):
        if self.cap is None:
            return

        nuevo_factor = self.combo_calidad.currentData()
        nombre_motor = self.combo_motor.currentData()
        carpeta_motor = os.path.join(CARPETA_BASE, "model_r", nombre_motor)

        ancho_original = self.ancho_original_corregido
        alto_original = self.alto_original_corregido

        ancho_bruto = int(ancho_original * nuevo_factor)
        alto_bruto = int(alto_original * nuevo_factor)
        nuevo_ancho = max(8, (ancho_bruto // 8) * 8)
        nuevo_alto = max(8, (alto_bruto // 8) * 8)

        print(f"Recargando RIFE con: {nombre_motor}, {nuevo_ancho}x{nuevo_alto}")
        try:
            rife_anterior = self.rife
            nuevo_rife = Rife(gpuid=0, model=carpeta_motor, width=nuevo_ancho, height=nuevo_alto)

            if hasattr(self, 'rife_worker') and self.rife_worker is not None:
                self.rife_worker.detener()
                self.rife_worker.wait(2000)

            self.rife = nuevo_rife
            self.ancho_final = nuevo_ancho
            self.alto_final = nuevo_alto

            while not self.cola_frames_listos.empty():
                try:
                    self.cola_frames_listos.get_nowait()
                except queue.Empty:
                    break

            self.rife_worker = RifeWorker(self.rife, self.cola_frames_originales, self.cola_frames_listos, self.fps_original, fps_salida=60)
            self.rife_worker.start()

            tiempo_audio_actual = self.audio_worker.obtener_posicion()
            self.numero_frame_mostrado = int(tiempo_audio_actual / self.tiempo_por_frame_interpolado)

            self._reconstruir_menus()
            self._guardar_configuracion()

            del rife_anterior
            gc.collect()
        except Exception as error:
            print(f"No se pudo aplicar esta calidad/motor: {error}")

    def _clic_en_barra_progreso(self, event):
        if self.cap is None:
            return
        proporcion = event.position().x() / self.barra_progreso.width()
        proporcion = max(0.0, min(1.0, proporcion))
        segundo_destino = proporcion * self.duracion_total_segundos
        self.saltar_a_posicion(segundo_destino)

    def saltar_relativo(self, segundos):
        if self.cap is None or not hasattr(self, 'audio_worker'):
            return
        tiempo_actual = self.audio_worker.obtener_posicion()
        nuevo_tiempo = max(0.0, min(self.duracion_total_segundos, tiempo_actual + segundos))
        self.saltar_a_posicion(nuevo_tiempo)     

    def saltar_a_posicion(self, segundo_destino):
        print("SALTO: iniciando")
        self.timer.stop()

        if hasattr(self, 'audio_worker') and self.audio_worker is not None:
            self.audio_worker.detener()
            self.audio_worker.wait(2000)
        if hasattr(self, 'decoder_worker') and self.decoder_worker is not None:
            self.decoder_worker.detener()
            self.decoder_worker.wait(2000)
        if hasattr(self, 'rife_worker') and self.rife_worker is not None:
            self.rife_worker.detener()
            self.rife_worker.wait(2000)

        while not self.cola_frames_originales.empty():
            try:
                self.cola_frames_originales.get_nowait()
            except queue.Empty:
                break
        while not self.cola_frames_listos.empty():
            try:
                self.cola_frames_listos.get_nowait()
            except queue.Empty:
                break

        if segundo_destino > 0:
            frame_inicio = int(segundo_destino * self.fps_original)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

        ret_prueba, _ = self.cap.read()
        segundo_real = segundo_destino

        self.decoder_worker = DecoderWorker(self.cap, self.cola_frames_originales, self.ancho_final, self.alto_final)
        self.decoder_worker.error.connect(lambda msg: print(f"ERROR DecoderWorker: {msg}"))
        self.decoder_worker.start()

        if self.interpolacion_disponible and self.checkbox_interpolacion.isChecked():
            self.rife_worker = RifeWorker(self.rife, self.cola_frames_originales, self.cola_frames_listos, self.fps_original, fps_salida=60)
            self.rife_worker.error.connect(lambda msg: print(f"ERROR RifeWorker: {msg}"))
            self.rife_worker.start()
        else:
            self.rife_worker = None

        self.audio_worker = AudioWorker(self.ruta_video_actual, segundo_inicio=segundo_real, indice_pista=self.indice_pista_audio_actual)
        self.audio_worker.error.connect(lambda msg: print(f"ERROR AudioWorker: {msg}"))
        self.audio_worker.start()

        self.numero_frame_mostrado = int(segundo_real / self.tiempo_por_frame_interpolado)
        self._debug_contador = 0
        self.timer.start(1)
        print("SALTO: timer reiniciado, terminado")

    def mostrar_siguiente_frame(self):
        cola_a_usar = self.cola_frames_listos if self.rife_worker is not None else self.cola_frames_originales

        try:
            item = cola_a_usar.get(timeout=0.01)
        except queue.Empty:
            self.timer.start(2)
            return

        if item is None:
            print("Video terminado")
            self.timer.stop()
            return

        if self.rife_worker is not None:
            frame_cv2 = item
        else:
            img_pil, frame_cv2 = item

        self.mostrar_frame_en_pantalla(frame_cv2)
        self.numero_frame_mostrado += 1

        if self.rife_worker is not None:
            tiempo_objetivo = self.numero_frame_mostrado * (self.tiempo_por_frame_interpolado)
        else:
            tiempo_objetivo = self.numero_frame_mostrado * (1 / self.fps_original)

        tiempo_audio_actual = self.audio_worker.obtener_posicion()
        diferencia = tiempo_objetivo - tiempo_audio_actual

        progreso_actual = min(1000, int((tiempo_audio_actual / self.duracion_total_segundos) * 1000))
        self.barra_progreso.setValue(progreso_actual)

        def formatear_tiempo(segundos_totales):
            horas = int(segundos_totales // 3600)
            minutos = int((segundos_totales % 3600) // 60)
            segundos = int(segundos_totales % 60)
            if horas > 0:
                return f"{horas}:{minutos:02d}:{segundos:02d}"
            else:
                return f"{minutos:02d}:{segundos:02d}"

        texto_actual = formatear_tiempo(tiempo_audio_actual)
        texto_total = formatear_tiempo(self.duracion_total_segundos)
        self.label_tiempo.setText(f"{texto_actual} / {texto_total}")

        # Lógica de sincronización que te funcionó
        if diferencia > 0.1: 
            espera = int(diferencia * 1000)
        elif diferencia < -0.1:
            espera = 1
        else:
            espera = 16
            
        self.timer.start(espera)

    def closeEvent(self, event):
        self.timer.stop()
        self.timer_rotacion_logo.stop()
        if hasattr(self, 'timer_animacion_carga'):
            self.timer_animacion_carga.stop()

        if hasattr(self, 'audio_worker') and self.audio_worker is not None:
            self.audio_worker.detener()
            self.audio_worker.wait(2000)

        if hasattr(self, 'hilo_carga') and self.hilo_carga.isRunning():
            self.hilo_carga.terminate()
            self.hilo_carga.wait()

        if hasattr(self, 'decoder_worker') and self.decoder_worker is not None:
            self.decoder_worker.detener()
            self.decoder_worker.wait(2000)

        if hasattr(self, 'rife_worker') and self.rife_worker is not None:
            self.rife_worker.detener()
            self.rife_worker.wait(2000)

        if self.subtitle_worker is not None:
            self.subtitle_worker.detener()
            self.subtitle_worker.wait(2000)

        if self.cap is not None:
            self.cap.release()

        event.accept()

QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
app = QApplication(sys.argv)
ventana = VentanaPrincipal()
ventana.show()
app.exec()


