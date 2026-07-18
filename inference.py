"""
flower_predict.py
Klasifikasi 5 jenis bunga + kendali 4 relay ESP32 via serial USB.

Alur singkat:
    pilih model -> buka koneksi relay -> upload foto -> prediksi -> relay
Tiap kelas bunga memetakan ke satu relay (toggle); kelas kelima memicu
seluruh relay. Pada mode komparasi, relay hanya aktif bila kedua model
sepakat dan tingkat keyakinannya melewati ambang.

Dependensi: tensorflow numpy matplotlib pyserial (tkinter opsional).
Sebelum dijalankan: unggah sketch ESP32, tutup Serial Monitor, lalu set PORT.
"""

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf


# --------------------------------------------------------------------------
# Pengaturan terpusat
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    classes: tuple = ("Aster", "Daisy", "Iris", "Lavender", "Lily")
    icons:   tuple = ("🌼", "🌸", "🪻", "💜", "🌷")
    side:    int = 224
    port:    str = "COM5"          # None untuk menonaktifkan relay
    baud:    int = 115200
    min_conf: float = 0.70
    here:    str = os.path.dirname(os.path.abspath(__file__))

    @property
    def cnn_file(self):
        return os.path.join(self.here, "cnn_best_model.keras")

    @property
    def tl_file(self):
        return os.path.join(self.here, "tl_best_model.keras")


CFG = Settings()

# Bunga -> nomor relay (1..4). "Lily" memakai aturan "semua relay".
RELAY_OF = {"Aster": 1, "Daisy": 2, "Iris": 3, "Lavender": 4}
ALL_RELAY_CLASS = "Lily"

PRETTY_NAME = {"cnn": "CNN From Scratch",
               "tl":  "MobileNetV2 Transfer Learning"}

# impor opsional ---------------------------------------------------------
try:
    import serial
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False

try:
    import matplotlib.pyplot as plt
    PLOT_OK = True
except ImportError:
    PLOT_OK = False

try:
    from tkinter import Tk, filedialog
    TK_OK = True
except ImportError:
    TK_OK = False


# --------------------------------------------------------------------------
# Lapisan serial ke ESP32
# --------------------------------------------------------------------------
class Esp32Link:
    """Pembungkus tipis di atas pyserial dengan pelacakan status relay."""

    def __init__(self):
        self._port = None
        self._toggled_all = False
        self._on = {n: False for n in range(1, 5)}
        self._dial_in()

    def _dial_in(self):
        if CFG.port is None:
            print("[relay] dimatikan (port = None).")
            return
        if not SERIAL_OK:
            print("[relay] modul pyserial tidak ada -> pip install pyserial")
            return
        try:
            self._port = serial.Serial(CFG.port, CFG.baud, timeout=1)
            time.sleep(2)                       # beri waktu ESP32 reboot
            print(f"[relay] tersambung: {CFG.port} @ {CFG.baud} baud")
        except Exception as err:
            print(f"[relay] tidak bisa membuka {CFG.port}: {err}")
            print("        pastikan Serial Monitor Arduino sudah ditutup.")

    def _raw(self, payload):
        if self._port is None:
            print(f"[relay/simulasi] -> {payload}")
            return
        try:
            self._port.write((payload + "\n").encode())
        except Exception as err:
            print(f"[relay] gagal menulis: {err}")

    # aturan per-kelas -----------------------------------------------------
    def trigger(self, flower):
        if flower == ALL_RELAY_CLASS:
            self._flip_everything()
        elif flower in RELAY_OF:
            self._flip_single(flower)

    def _flip_everything(self):
        self._toggled_all = not self._toggled_all
        self._raw("A1" if self._toggled_all else "A0")
        for n in self._on:
            self._on[n] = self._toggled_all
        print(f"[relay] {ALL_RELAY_CLASS}: semua "
              f"{'menyala' if self._toggled_all else 'mati'}")

    def _flip_single(self, flower):
        n = RELAY_OF[flower]
        self._on[n] = not self._on[n]
        self._raw(f"{n}{'1' if self._on[n] else '0'}")
        print(f"[relay] {flower}: relay {n} "
              f"{'menyala' if self._on[n] else 'mati'}")

    def reset(self):
        self._raw("A0")
        self._toggled_all = False
        self._on = {n: False for n in range(1, 5)}

    def shutdown(self):
        if self._port is not None:
            self.reset()
            self._port.close()


# --------------------------------------------------------------------------
# Manajemen model
# --------------------------------------------------------------------------
class ModelBank:
    """Memuat model sekali lalu menyimpannya (cache)."""

    def __init__(self):
        self._cache = {}

    def fetch(self, key):
        if key in self._cache:
            return self._cache[key]
        path = CFG.cnn_file if key == "cnn" else CFG.tl_file
        if not os.path.isfile(path):
            sys.exit(f"[error] berkas model hilang: {path}")
        print(f"memuat {PRETTY_NAME[key]} ...")
        self._cache[key] = tf.keras.models.load_model(path, compile=False)
        print("   ok.")
        return self._cache[key]


# --------------------------------------------------------------------------
# Inti prediksi
# --------------------------------------------------------------------------
def _as_batch(rgb):
    resized = tf.image.resize(rgb, (CFG.side, CFG.side))
    return tf.expand_dims(tf.cast(resized, tf.float32), 0)


def _run_single(model, rgb):
    vec = model.predict(_as_batch(rgb), verbose=0)[0]
    top = int(np.argmax(vec))
    return {
        "name": CFG.classes[top],
        "icon": CFG.icons[top],
        "conf": float(vec[top]),
        "vec":  vec,
    }


def run_models(bank_models, rgb):
    """bank_models: dict {key: model}. Kembalikan hasil tiap model."""
    out = {k: _run_single(m, rgb) for k, m in bank_models.items()}
    if {"cnn", "tl"} <= out.keys():
        out["match"] = out["cnn"]["name"] == out["tl"]["name"]
    return out


def relay_target(out, keys):
    """Pilih (kelas, conf) untuk relay; (None, 0) bila tidak memenuhi syarat."""
    if len(keys) == 1:
        r = out[keys[0]]
        return (r["name"], r["conf"]) if r["conf"] >= CFG.min_conf else (None, 0)
    if out.get("match"):
        c = min(out["cnn"]["conf"], out["tl"]["conf"])
        if c >= CFG.min_conf:
            return out["cnn"]["name"], c
    return None, 0


# --------------------------------------------------------------------------
# Output ke layar
# --------------------------------------------------------------------------
def echo_result(out, keys):
    line = "-" * 48
    print(line)
    for k in keys:
        r = out[k]
        print(f"{PRETTY_NAME[k]:<32}: {r['icon']} {r['name']} "
              f"({r['conf'] * 100:.1f}%)")
    if "match" in out:
        print("=> sepakat" if out["match"] else "=> berbeda")
    print(line)


def draw_figure(rgb, out, keys, src):
    if not PLOT_OK:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5),
                                      gridspec_kw={"width_ratios": [1, 1.3]})
    left.imshow(rgb)
    left.axis("off")
    cap = []
    for k in keys:
        r = out[k]
        cap.append(f"{k.upper()}: {r['icon']} {r['name']} ({r['conf']*100:.0f}%)")
    if "match" in out:
        cap.append("sepakat" if out["match"] else "berbeda")
    left.set_title("\n".join(cap), fontsize=11, loc="left")

    xs = np.arange(len(CFG.classes))
    palette = {"cnn": "#4C8BF5", "tl": "#F5A623"}
    if len(keys) == 1:
        k = keys[0]
        right.bar(xs, out[k]["vec"] * 100, 0.6,
                  color=palette[k], label=PRETTY_NAME[k])
    else:
        w = 0.38
        right.bar(xs - w / 2, out["cnn"]["vec"] * 100, w,
                  color=palette["cnn"], label="CNN")
        right.bar(xs + w / 2, out["tl"]["vec"] * 100, w,
                  color=palette["tl"], label="Transfer")
    right.set_xticks(xs)
    right.set_xticklabels(CFG.classes, rotation=20, ha="right")
    right.set_ylim(0, 105)
    right.set_ylabel("Probabilitas (%)")
    right.grid(axis="y", alpha=0.3)
    right.legend()
    right.set_title("Distribusi probabilitas")

    fig.suptitle(f"Hasil — {os.path.basename(src)}", fontweight="bold")
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# Mode upload foto
# --------------------------------------------------------------------------
def _ask_path():
    if TK_OK:
        print("\nbuka dialog berkas ...")
        root = Tk()
        root.withdraw()
        chosen = filedialog.askopenfilename(
            title="Pilih gambar bunga",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp *.webp")])
        root.destroy()
        return chosen or ""
    return input("path gambar: ").strip().strip('"')


def _decode(path):
    blob = tf.io.read_file(path)
    arr = tf.image.decode_image(blob, channels=3, expand_animations=False)
    return arr.numpy().astype("uint8")


def upload_flow(active, link):
    keys = list(active.keys())
    path = _ask_path()
    if not path:
        print("dibatalkan.")
        return
    if not os.path.isfile(path):
        print(f"berkas tidak ada: {path}")
        return

    rgb = _decode(path)
    print(f"memproses {os.path.basename(path)} ...")
    out = run_models(active, rgb)
    echo_result(out, keys)

    flower, _ = relay_target(out, keys)
    if flower:
        link.trigger(flower)
    else:
        print("[relay] tidak aktif (keyakinan kurang / model tak sepakat)")

    draw_figure(rgb, out, keys, path)


# --------------------------------------------------------------------------
# Antarmuka teks
# --------------------------------------------------------------------------
def choose_models(bank):
    prompt = ("\nMODEL\n"
              "  1) CNN From Scratch\n"
              "  2) MobileNetV2 Transfer Learning\n"
              "  3) Komparasi keduanya\n"
              "pilih (1/2/3): ")
    while True:
        pick = input(prompt).strip()
        if pick == "1":
            return {"cnn": bank.fetch("cnn")}
        if pick == "2":
            return {"tl": bank.fetch("tl")}
        if pick == "3":
            return {"cnn": bank.fetch("cnn"), "tl": bank.fetch("tl")}
        print("masukan tidak dikenal.")


def header(active):
    mode = ("Komparasi CNN vs Transfer Learning"
            if len(active) == 2 else PRETTY_NAME[next(iter(active))])
    print("\n" + "#" * 48)
    print("  KLASIFIKASI BUNGA + RELAY ESP32")
    print(f"  mode: {mode}")
    print("#" * 48)
    print("  kelas : 🌼Aster 🌸Daisy 🪻Iris 💜Lavender 🌷Lily")
    print("  relay : Aster=1 Daisy=2 Iris=3 Lavender=4 | Lily=semua")
    print("-" * 48)
    print("  1) Upload foto")
    print("  2) Ganti model")
    print("  3) Keluar")


def run():
    print("inisialisasi ...")
    bank = ModelBank()
    active = choose_models(bank)
    link = Esp32Link()
    try:
        while True:
            header(active)
            pick = input("pilih (1/2/3): ").strip()
            if pick == "1":
                upload_flow(active, link)
            elif pick == "2":
                active = choose_models(bank)
            elif pick == "3":
                print("selesai.")
                break
            else:
                print("masukan tidak dikenal.")
    finally:
        link.shutdown()
        print("relay dimatikan, port ditutup.")


if __name__ == "__main__":
    run()
