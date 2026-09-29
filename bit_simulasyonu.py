import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Fotografi oku ve gri tona cevir
img_bgr = cv2.imread('insan.jpg')

# Eger resim bulunamazsa test icin otomatik gri resim olustur
if img_bgr is None:
    print("Uyari: 'insan.jpg' bulunamadi, test icin ornek gri resim olusturuluyor.")
    img_gray = np.tile(np.linspace(0, 255, 300, dtype=np.uint8), (300, 1))
else:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

yukseklik, genislik = img_gray.shape

# Islenmis yeni resimler icin bos kopyalar
img_yari = np.zeros((yukseklik, genislik), dtype=np.uint8)
img_ceyrek = np.zeros((yukseklik, genislik), dtype=np.uint8)
img_128 = np.zeros((yukseklik, genislik), dtype=np.uint8)
img_6bit = np.zeros((yukseklik, genislik), dtype=np.uint8)

# 2. Satir ve sutunlari gezerek bit seviyelerini simule et
for i in range(yukseklik):
    for j in range(genislik):
        p = int(img_gray[i, j])
        
        # Parlaklik degerlerini yariya ve ceyrege bol
        img_yari[i, j] = p // 2
        img_ceyrek[i, j] = p // 4
        
        # 128 farkli deger (7-bit simulasyonu)
        img_128[i, j] = (p // 2) * 2
        
        # 64 farkli deger (6-bit simulasyonu)
        img_6bit[i, j] = (p // 4) * 4

# 3. Sonuclari gorsellestirme
basliklar = [
    'Orijinal Gri (8-bit / 256 Seviye)', 
    'Parlaklik Yariya Bolundu (// 2)', 
    'Parlaklik Ceyrege Bolundu (// 4)', 
    '128 Farkli Deger ((p//2)*2)', 
    '6-bit / 64 Deger ((p//4)*4)'
]
resimler = [img_gray, img_yari, img_ceyrek, img_128, img_6bit]

plt.figure(figsize=(15, 8))
for k in range(5):
    plt.subplot(2, 3, k + 1)
    plt.imshow(resimler[k], cmap='gray', vmin=0, vmax=255)
    plt.title(basliklar[k], fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()