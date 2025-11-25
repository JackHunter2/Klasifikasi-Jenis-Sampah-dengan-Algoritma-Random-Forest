/**
 * Validasi file sebelum upload
 * @returns {boolean} true jika file valid, false jika tidak
 */
function validateFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Silakan pilih file gambar terlebih dahulu!');
        return false;
    }
    
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        alert('Format file tidak didukung! Gunakan: JPG, PNG, GIF, atau BMP');
        return false;
    }
    
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        alert('Ukuran file terlalu besar! Maksimal 10MB');
        return false;
    }
    
    return true;
}

