import sys
import cv2
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from BT import Ui_MainWindow 

class FA(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        self.cap = None  
        self.timer = QtCore.QTimer()  
        self.timer.timeout.connect(self.update_frame)

        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop.clicked.connect(self.stop_detection)

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)  

    def stop_detection(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.label_video.clear()
        self.label_face_count.setText("Face: 0")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_count = len(faces)
        self.label_face_count.setText(f"Face: {face_count}")

        x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = x.shape
        b = ch * w
        cv = QImage(x.data, w, h, b, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(cv))

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FA()
    window.show()
    sys.exit(app.exec())
