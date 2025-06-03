import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QDialog, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
import io
from PIL import Image
import torch
import numpy as np
from model import load_image, RRDBNet

class WelcomeWindow(QtWidgets.QMainWindow):
    def __init__(self, app):
        super().__init__()
        uic.loadUi('ui/welcome_screen.ui', self)
        self.app = app
        self.pushButton.clicked.connect(self.app.show_main_screen)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app):
        super().__init__()
        uic.loadUi('ui/main_screen.ui', self)
        self.app = app
        self.pushButton.clicked.connect(self.upload_image)
        self.pushButton_1.clicked.connect(self.show_depixel_screen)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphicsView.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def upload_image(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Image File', '',
                                                             "Image files (*.jpg *.jpeg *.png)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.scene.clear()  # Clear previous image
            item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(item)
            # Use individual properties of QRect to set the scene rectangle
            self.scene.setSceneRect(pixmap.rect().x(), pixmap.rect().y(), pixmap.rect().width(), pixmap.rect().height())
            self.update_view()
            self.app.current_image_path = file_name  # Store the image path for later use

    def update_view(self):
        self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)  # Fit the scene in the view

    def show_depixel_screen(self):
        self.app.show_depixel_screen()

    def wheelEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        is_ctrl_pressed = modifiers == Qt.ControlModifier  # Check if the control key is pressed

        # Handle zoom only if Control is pressed (common convention in many applications)
        if is_ctrl_pressed:
            delta = event.angleDelta().y() / 120  # Divide by 120 to normalize the delta
            factor = pow(1.1, delta)  # Calculate the zoom factor

            # Apply the zoom factor while ensuring the zoom center is under the mouse cursor
            self.graphicsView.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.graphicsView.scale(factor, factor)
            self.graphicsView.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        else:
            # Handle normal scrolling by translating the scene
            delta_x = -event.angleDelta().x() / 5  # Normalize and invert x delta for horizontal scroll
            delta_y = -event.angleDelta().y() / 5  # Normalize and invert y delta for vertical scroll
            self.graphicsView.translate(delta_x, delta_y)

        event.accept()  # Accept the event to prevent propagation

class WaitDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processing")
        self.setModal(True)
        layout = QVBoxLayout()
        label = QLabel("Будь ласка, зачекайте, йде обробка зображення...")
        layout.addWidget(label)
        self.setLayout(layout)

class DepixelWindow(QtWidgets.QMainWindow):
    def __init__(self, app):
        super().__init__()
        uic.loadUi('ui/depixel_screen.ui', self)
        self.app = app
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphicsView.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.pushButton.clicked.connect(self.app.show_main_screen)
        self.pushButton_1.clicked.connect(self.show_compare_screen)
        self.pushButton_2.clicked.connect(self.save_depixel_image)  # Connect the save button

    def showEvent(self, event):
        # Automatically process the image when this screen is shown
        if self.app.current_image_path and not self.app.depixel_image:
            self.process_image()
        elif self.app.depixel_image:
            self.display_image(self.app.depixel_image)

    def process_image(self):
        self.wait_dialog = WaitDialog()
        self.wait_dialog.show()
        QtWidgets.QApplication.processEvents()  # Ensure the dialog is shown

        # Load and process the image
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
        model.load_state_dict(torch.load('BSRGAN.pth'), strict=False)
        model.eval()

        image = load_image(self.app.current_image_path)
        if image:
            output_image = self.run_depixelation(model, image)
            self.app.depixel_image = output_image  # Store the depixelated image for later use
            self.display_image(output_image)

        self.wait_dialog.close()

    def run_depixelation(self, model, image):
        # Convert the PIL Image to a numpy array
        image_np = np.array(image).astype(np.float32)

        # Normalize the image data to 0-1
        image_np /= 255.0

        # Convert the numpy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(
             0)  # Permute to move the channel to the first dimension

        # Check and move the model to CPU (assuming no GPU support in the console environment)
        device = torch.device('cpu')
        model.to(device)
        image_tensor = image_tensor.to(device)

        # Apply the model in no_grad context to prevent tracking gradients which we don't need in inference
        with torch.no_grad():
            output_tensor = model(image_tensor)

        # Convert the output tensor back to an image
        # Squeeze to remove the batch dimension, permute to put the channels back to the last position, and clamp the values to ensure they are within [0,1] after possible processing overshoot
        output_image_np = output_tensor.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()

        # Denormalize by converting back to 0-255 range and cast to unsigned 8-bit integer
        output_image_np = (output_image_np * 255).astype(np.uint8)

        # Convert numpy array back to a PIL Image
        output_image = Image.fromarray(output_image_np)
        return output_image

    def display_image(self, image):
        # Convert PIL image to QPixmap
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        pixmap = QPixmap()
        pixmap.loadFromData(byte_arr)

        # Display the QPixmap in the QGraphicsView
        self.scene.clear()
        item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.scene.setSceneRect(pixmap.rect().x(), pixmap.rect().y(), pixmap.rect().width(), pixmap.rect().height())
        self.update_view()

    def update_view(self):
        self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)  # Fit the scene in the view

    def wheelEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        is_ctrl_pressed = modifiers == Qt.ControlModifier  # Check if the control key is pressed

        # Handle zoom only if Control is pressed (common convention in many applications)
        if is_ctrl_pressed:
            delta = event.angleDelta().y() / 120  # Divide by 120 to normalize the delta
            factor = pow(1.1, delta)  # Calculate the zoom factor

            # Apply the zoom factor while ensuring the zoom center is under the mouse cursor
            self.graphicsView.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.graphicsView.scale(factor, factor)
            self.graphicsView.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        else:
            # Handle normal scrolling by translating the scene
            delta_x = -event.angleDelta().x() / 5  # Normalize and invert x delta for horizontal scroll
            delta_y = -event.angleDelta().y() / 5  # Normalize and invert y delta for vertical scroll
            self.graphicsView.translate(delta_x, delta_y)

        event.accept()  # Accept the event to prevent propagation

    def show_compare_screen(self):
        self.app.show_compare_screen()

    def save_depixel_image(self):
        if self.app.depixel_image:
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Depixelated Image", "",
                                                                 "PNG files (*.png);;JPG files (*.jpg);;JPEG files (*.jpeg)")
            if file_path:
                self.app.depixel_image.save(file_path)

class CompareWindow(QtWidgets.QMainWindow):
    def __init__(self, app):
        super().__init__()
        uic.loadUi('ui/compare_screen.ui', self)
        self.app = app
        self.scene_original = QGraphicsScene()
        self.scene_depixel = QGraphicsScene()
        self.graphicsView.setScene(self.scene_original)
        self.graphicsView_1.setScene(self.scene_depixel)
        self.pushButton.clicked.connect(self.show_depixel_screen)

    def showEvent(self, event):
        if self.app.current_image_path and self.app.depixel_image:
            self.display_images()

    def display_images(self):
        # Display original image
        pixmap_original = QPixmap(self.app.current_image_path)
        self.scene_original.clear()
        item_original = QGraphicsPixmapItem(pixmap_original)
        self.scene_original.addItem(item_original)
        self.scene_original.setSceneRect(pixmap_original.rect().x(), pixmap_original.rect().y(),
                                         pixmap_original.rect().width(), pixmap_original.rect().height())
        self.graphicsView.fitInView(self.scene_original.sceneRect(), Qt.KeepAspectRatio)

        # Display depixelated image
        byte_arr = io.BytesIO()
        self.app.depixel_image.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        pixmap_depixel = QPixmap()
        pixmap_depixel.loadFromData(byte_arr)
        self.scene_depixel.clear()
        item_depixel = QGraphicsPixmapItem(pixmap_depixel)
        self.scene_depixel.addItem(item_depixel)
        self.scene_depixel.setSceneRect(pixmap_depixel.rect().x(), pixmap_depixel.rect().y(),
                                        pixmap_depixel.rect().width(), pixmap_depixel.rect().height())
        self.graphicsView_1.fitInView(self.scene_depixel.sceneRect(), Qt.KeepAspectRatio)

    def show_depixel_screen(self):
        self.app.show_depixel_screen()

    def wheelEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        is_ctrl_pressed = modifiers == Qt.ControlModifier  # Check if the control key is pressed

        # Determine which graphics view the cursor is over
        cursor_pos = event.globalPos()
        graphics_view_under_cursor = None
        if self.graphicsView.underMouse():
            graphics_view_under_cursor = self.graphicsView
        elif self.graphicsView_1.underMouse():
            graphics_view_under_cursor = self.graphicsView_1

        if graphics_view_under_cursor:
            if is_ctrl_pressed:
                delta = event.angleDelta().y() / 120  # Divide by 120 to normalize the delta
                factor = pow(1.1, delta)  # Calculate the zoom factor

                # Apply the zoom factor while ensuring the zoom center is under the mouse cursor
                graphics_view_under_cursor.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
                graphics_view_under_cursor.scale(factor, factor)
                graphics_view_under_cursor.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            else:
                # Handle normal scrolling by translating the scene
                delta_x = -event.angleDelta().x() / 5  # Normalize and invert x delta for horizontal scroll
                delta_y = -event.angleDelta().y() / 5  # Normalize and invert y delta for vertical scroll
                graphics_view_under_cursor.translate(delta_x, delta_y)

            event.accept()  # Accept the event to prevent propagation

class App(QtWidgets.QApplication):
    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)
        self.welcome_window = WelcomeWindow(self)
        self.main_window = MainWindow(self)
        self.depixel_window = DepixelWindow(self)
        self.compare_window = CompareWindow(self)
        self.current_image_path = None  # Variable to store the path of the currently uploaded image
        self.depixel_image = None  # Variable to store the depixelated image

    def show_welcome_screen(self):
        self.main_window.hide()
        self.depixel_window.hide()
        self.compare_window.hide()
        self.welcome_window.show()

    def show_main_screen(self):
        self.welcome_window.hide()
        self.depixel_window.hide()
        self.compare_window.hide()
        self.main_window.show()

    def show_depixel_screen(self):
        self.main_window.hide()
        self.welcome_window.hide()
        self.compare_window.hide()
        self.depixel_window.show()

    def show_compare_screen(self):
        self.depixel_window.hide()
        self.compare_window.show()

if __name__ == '__main__':
    app = App(sys.argv)
    app.show_welcome_screen()
    sys.exit(app.exec_())

