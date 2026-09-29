import cv2
import numpy as np
import time

# 1. Fotografi oku ve griye cevir
img_bgr = cv2.imread('insan.jpg')
if img_bgr is None:
    raise FileNotFoundError("'insan.jpg' dosyasi bulunamadi! Klasore resmi ekleyin.")

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
yukseklik, genislik = img_gray.shape

print(f"Resim Boyutu: {genislik}x{yukseklik} ({genislik * yukseklik:,} piksel)\n")

cikis1 = np.zeros_like(img_gray)
cikis2 = np.zeros_like(img_gray)

# DENEY 1: Once Satirlar (i), Sonra Sutunlar (j) - [Row-Major: Hizli Olan]
baslangic1 = time.perf_counter()
for i in range(yukseklik):
    for j in range(genislik):
        p = int(img_gray[i, j])
        cikis1[i, j] = (p // 4) * 4
bitis1 = time.perf_counter()
sure1 = bitis1 - baslangic1
print(f"1. Yontem (Satir -> Sutun) Suresi : {sure1:.4f} saniye")

# DENEY 2: Once Sutunlar (j), Sonra Satirlar (i) - [Column-Major: Yavas Olan]
baslangic2 = time.perf_counter()
for j in range(genislik):
    for i in range(yukseklik):
        p = int(img_gray[i, j])
        cikis2[i, j] = (p // 4) * 4
bitis2 = time.perf_counter()
sure2 = bitis2 - baslangic2
print(f"2. Yontem (Sutun -> Satir) Suresi : {sure2:.4f} saniye")

# Kiyaslama
oran = (sure2 / sure1) if sure1 > 0 else 0
print("\n--- Degerlendirme ---")
print(f"Satir-Sutun sirasi, Sutun-Satir sirasina gore {oran:.2f} kat daha hizli calisti.")