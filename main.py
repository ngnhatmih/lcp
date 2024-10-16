from license_plate_detection import detect_license_plate

file = '0056_01595_b.jpg'
plate_number = detect_license_plate(file)
print("Biển số xe:", plate_number)