import cv2
import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt

def piksel_donustur(roi):
    h, w = roi.shape
    islenmis_roi = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            p = float(roi[i, j])
            yeni_p = int(p * 0.75 + 20)
            
            if yeni_p > 255:
                yeni_p = 255
            elif yeni_p < 0:
                yeni_p = 0
                
            islenmis_roi[i, j] = yeni_p
            
    return islenmis_roi

if __name__ == '__main__':
    img_bgr = cv2.imread('insan.jpg')
    if img_bgr is None:
        raise FileNotFoundError("'insan.jpg' bulunamadi!")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Islemcinin farki hissetmesi icin boyutlandirma
    img_gray = cv2.resize(img_gray, (3000, 3000))
    h, w = img_gray.shape

    orta_y, orta_x = h // 2, w // 2
    roiler = [
        img_gray[0:orta_y, 0:orta_x],
        img_gray[0:orta_y, orta_x:w],
        img_gray[orta_y:h, 0:orta_x],
        img_gray[orta_y:h, orta_x:w]
    ]

    # 1. Tek Cekirdek
    basla_tek = time.perf_counter()
    sonuclar_tek = [piksel_donustur(r) for r in roiler]
    bitir_tek = time.perf_counter()
    sure_tek = bitir_tek - basla_tek

    # 2. 4 Cekirdek Paralel
    basla_paralel = time.perf_counter()
    with Pool(processes=4) as pool:
        sonuclar_paralel = pool.map(piksel_donustur, roiler)
    bitir_paralel = time.perf_counter()
    sure_paralel = bitir_paralel - basla_paralel

    hizlanma = sure_tek / sure_paralel if sure_paralel > 0 else 0
    print("=" * 50)
    print("       (p * 0.75 + 20) PERFORMANS RAPORU")
    print("=" * 50)
    print(f"Tek Cekirdek Suresi       : {sure_tek:.4f} saniye")
    print(f"4 Cekirdek Paralel Suresi : {sure_paralel:.4f} saniye")
    print(f"Hizlanma Orani (Speedup)  : {hizlanma:.2f} kat daha hizli!")
    print("=" * 50)

    ust = np.hstack((sonuclar_paralel[0], sonuclar_paralel[1]))
    alt = np.hstack((sonuclar_paralel[2], sonuclar_paralel[3]))
    nihai_resim = np.vstack((ust, alt))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("Orijinal Gri Resim")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(nihai_resim, cmap='gray', vmin=0, vmax=255)
    plt.title("Donusturulmus Resim (0.75 * p + 20)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()