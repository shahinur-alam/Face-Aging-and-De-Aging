import sys
import torch
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from diffusers import StableDiffusionImg2ImgPipeline


class FaceAgingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()

    def initUI(self):
        self.setWindowTitle('AI Face Aging App (Diffusion Model)')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.imageLabel)

        loadButton = QPushButton('Load Image', self)
        loadButton.clicked.connect(self.loadImage)
        layout.addWidget(loadButton)

        self.ageSlider = QSlider(Qt.Horizontal)
        self.ageSlider.setMinimum(20)
        self.ageSlider.setMaximum(70)
        self.ageSlider.setValue(20)
        self.ageSlider.setTickPosition(QSlider.TicksBelow)
        self.ageSlider.setTickInterval(10)
        self.ageSlider.valueChanged.connect(self.updateAge)
        layout.addWidget(self.ageSlider)

        self.ageLabel = QLabel('Age: 20', self)
        self.ageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.ageLabel)

        processButton = QPushButton('Process', self)
        processButton.clicked.connect(self.processImage)
        layout.addWidget(processButton)

        self.setLayout(layout)

    def loadModel(self):
        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def loadImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if fileName:
            self.image = Image.open(fileName).convert('RGB')
            self.displayImage(self.image)

    def displayImage(self, image):
        qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def updateAge(self, value):
        self.ageLabel.setText(f'Age: {value}')

    def processImage(self):
        if not hasattr(self, 'image'):
            return

        target_age = self.ageSlider.value()

        # Prepare prompt
        prompt = f"A photo of a person aged {target_age} years old, same person, realistic"

        # Process image
        with torch.no_grad():
            aged_image = self.pipe(prompt=prompt, image=self.image, strength=0.75, guidance_scale=7.5).images[0]

        # Display result
        self.displayImage(aged_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceAgingApp()
    ex.show()
    sys.exit(app.exec_())