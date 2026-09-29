import cv2
import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt

def roi_histogram_hesapla(roi):
    h, w = roi.shape
    yerel_histogram = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            piksel_degeri = int(roi[i, j])
            yerel_histogram[piksel_degeri] += 1
    return yerel_histogram

if __name__ == '__main__':
    img_bgr = cv2.imread('insan.jpg')
    if img_bgr is None:
        raise FileNotFoundError("'insan.jpg' bulunamadi!")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    yukseklik, genislik = img_gray.shape
    toplam_piksel = yukseklik * genislik

    orta_y = yukseklik // 2
    orta_x = genislik // 2

    roiler = [
        img_gray[0:orta_y, 0:orta_x],
        img_gray[0:orta_y, orta_x:genislik],
        img_gray[orta_y:yukseklik, 0:orta_x],
        img_gray[orta_y:yukseklik, orta_x:genislik]
    ]

    baslangic = time.perf_counter()
    with Pool(processes=4) as pool:
        parca_histogramlari = pool.map(roi_histogram_hesapla, roiler)
    bitis = time.perf_counter()

    genel_histogram = (parca_histogramlari[0] + 
                       parca_histogramlari[1] + 
                       parca_histogramlari[2] + 
                       parca_histogramlari[3])

    sayilan_toplam_piksel = np.sum(genel_histogram)
    print("=" * 45)
    print("       PARALEL HISTOGRAM RAPORU")
    print("=" * 45)
    print(f"Resim Boyutu            : {genislik}x{yukseklik} ({toplam_piksel} piksel)")
    print(f"Histogram Toplam Piksel : {sayilan_toplam_piksel}")
    print(f"Hesaplama Suresi        : {bitis - baslangic:.4f} saniye")
    print(f"Dogrulama               : {'BASARILI' if toplam_piksel == sayilan_toplam_piksel else 'HATALI'}")
    print("=" * 45)

    plt.figure(figsize=(15, 8))
    etiketler = ['1. Parca (Sol-Ust)', '2. Parca (Sag-Ust)', '3. Parca (Sol-Alt)', '4. Parca (Sag-Alt)']
    renkler = ['blue', 'green', 'orange', 'purple']

    for idx in range(4):
        plt.subplot(2, 3, idx + 1)
        plt.bar(range(256), parca_histogramlari[idx], color=renkler[idx], width=1.0)
        plt.title(f"{etiketler[idx]} Histogrami")
        plt.xlabel("Parlaklik Degeri (0-255)")
        plt.ylabel("Piksel Sayisi")
        plt.xlim([0, 255])

    plt.subplot(2, 3, 5)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Orijinal Gorsel")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.bar(range(256), genel_histogram, color='black', width=1.0)
    plt.title("Genel Toplam Histogram")
    plt.xlabel("Parlaklik Degeri (0-255)")
    plt.ylabel("Piksel Sayisi")
    plt.xlim([0, 255])

    plt.tight_layout()
    plt.show()