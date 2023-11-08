import cv2

# Inisialisasi classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Baca gambar JPG
image = cv2.imread('gambar.jpg')

# Konversi gambar ke skala abu-abu (grayscale) untuk deteksi wajah
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Lakukan deteksi wajah
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Gambar kotak di sekitar wajah yang terdeteksi
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Menyimpan gambar yang telah diubah
cv2.imwrite('gambar_dengan_wajah.jpg', image)

# Tampilkan gambar dengan wajah yang telah terdeteksi
cv2.imshow('Deteksi Wajah', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
