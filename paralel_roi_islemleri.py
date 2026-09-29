import cv2
import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt

def roi_isle(veri):
    roi, bolum_no = veri
    h, w = roi.shape
    islenmis_roi = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            p = int(roi[i, j])
            if bolum_no == 1:
                islenmis_roi[i, j] = p // 2          # Parlaklik yariya
            elif bolum_no == 2:
                islenmis_roi[i, j] = p // 4          # Parlaklik ceyrege
            elif bolum_no == 3:
                islenmis_roi[i, j] = (p // 16) * 16  # 4-bit (16 seviye)
            else:
                islenmis_roi[i, j] = (p // 64) * 64  # 2-bit (4 seviye)
                
    return islenmis_roi

if __name__ == '__main__':
    img_bgr = cv2.imread('insan.jpg')
    if img_bgr is None:
        raise FileNotFoundError("'insan.jpg' bulunamadi!")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    yukseklik, genislik = img_gray.shape

    orta_y = yukseklik // 2
    orta_x = genislik // 2

    roi1 = img_gray[0:orta_y, 0:orta_x]
    roi2 = img_gray[0:orta_y, orta_x:genislik]
    roi3 = img_gray[orta_y:yukseklik, 0:orta_x]
    roi4 = img_gray[orta_y:yukseklik, orta_x:genislik]

    gorevler = [(roi1, 1), (roi2, 2), (roi3, 3), (roi4, 4)]

    # 1. Tek Cekirdek (Sirali)
    baslangic_tek = time.perf_counter()
    sonuclar_tek = [roi_isle(g) for g in gorevler]
    bitis_tek = time.perf_counter()
    sure_tek = bitis_tek - baslangic_tek

    # 2. 4 Cekirdek (Paralel)
    baslangic_paralel = time.perf_counter()
    with Pool(processes=4) as pool:
        sonuclar_paralel = pool.map(roi_isle, gorevler)
    bitis_paralel = time.perf_counter()
    sure_paralel = bitis_paralel - baslangic_paralel

    hizlanma = sure_tek / sure_paralel if sure_paralel > 0 else 0
    print("=" * 45)
    print("           ZAMAN OLCUM RAPORU")
    print("=" * 45)
    print(f"Tek Cekirdek (Sirali) Sure : {sure_tek:.4f} saniye")
    print(f"4 Cekirdek (Paralel) Sure  : {sure_paralel:.4f} saniye")
    print(f"Hizlanma Orani (Speedup)   : {hizlanma:.2f} kat daha hizli")
    print("=" * 45)

    ust = np.hstack((sonuclar_paralel[0], sonuclar_paralel[1]))
    alt = np.hstack((sonuclar_paralel[2], sonuclar_paralel[3]))
    nihai_resim = np.vstack((ust, alt))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Orijinal Resim")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(nihai_resim, cmap='gray')
    plt.title("4 Cekirdekle Paralel Islenen ROI'ler")
    plt.axis('off')

    plt.tight_layout()
    plt.show()