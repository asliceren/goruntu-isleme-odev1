# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 1. Resmi Gri Tonlamali Olarak Yukleme
input_path = "dusuk_kontrast.jpg"
image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Hata: Resim bulunamadi! Dosya adini kontrol edin.")
    exit()

height, width = image.shape
total_pixels = height * width

# 2. Histogram Hesaplama (0-255 arasi piksel frekanslari)
histogram = [0] * 256
for y in range(height):
    for x in range(width):
        pixel_val = image[y, x]
        histogram[pixel_val] += 1

# 3. Kumulatif Dagilim Fonksiyonu (CDF) Hesaplama
cdf = [0] * 256
cumulative_sum = 0
for i in range(256):
    cumulative_sum += histogram[i]
    cdf[i] = cumulative_sum

# Sifirdan buyuk ilk CDF degerini bulma
cdf_min = 0
for val in cdf:
    if val > 0:
        cdf_min = val
        break

# 4. Lookup Table (LUT) Olusturma
lut = [0] * 256
if total_pixels - cdf_min == 0:
    lut = list(range(256))
else:
    for v in range(256):
        if cdf[v] < cdf_min:
            lut[v] = 0
        else:
            mapped_val = round(((cdf[v] - cdf_min) / (total_pixels - cdf_min)) * 255)
            lut[v] = int(np.clip(mapped_val, 0, 255))

# 5. Yeni Esitlenmis Resmi Uretme
esitlenmis_resim = np.zeros((height, width), dtype=np.uint8)
for y in range(height):
    for x in range(width):
        eski_piksel = image[y, x]
        esitlenmis_resim[y, x] = lut[eski_piksel]

# 6. Cikti Resmini Diske Kaydetme
output_path = "esitlenmis_cikis.jpg"
cv2.imwrite(output_path, esitlenmis_resim)
print("Islem tamamlandi. Yeni resim kaydedildi: " + output_path)

# 7. Resimleri Ekranda Sabit Gosterme
cv2.imshow("Orijinal Resim", image)
cv2.imshow("Esitlenmis Resim", esitlenmis_resim)

# Pencere acikken herhangi bir tusa basilana kadar bekle
while True:
    k = cv2.waitKey(100) & 0xFF
    if k != 255:
        break

cv2.destroyAllWindows()