import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QLineEdit, QListView,
                             QMessageBox, QVBoxLayout, QLabel, QDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from wordcloud import WordCloud

from chat_log import ChatLogModel, DrawSpeechBubbleDelegate

from models.generating_model import GeneratingModel
from models.emotions_detection_custom import MultipleEmotionsClassifier

style_sheet = """ 
    QPushButton {
        background: #83E56C /* Green */
    }
    QListView {
        background: #FDF3DD
    }"""

greetings_message = "Hello. I'm bot that checks your emotional state. " \
                    "We will have a little conversation."


class WordCloudDialog(QDialog):
    def __init__(self, frequencies, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Your results')
        self.ok_button = QPushButton("Ok")
        self.ok_button.clicked.connect(self.close)

        wc = WordCloud(width=800, height=400,
                       background_color='white') \
            .generate_from_frequencies(frequencies)
        img = np.array(wc.to_image())
        height, width, byte_val = img.shape
        byte_val = byte_val * width
        image = QImage(img.data, width, height, byte_val, QImage.Format_RGB888)
        pixmap = QPixmap(image)
        self.image_lbl = QLabel()
        self.image_lbl.setPixmap(QPixmap(pixmap))
        lay = QVBoxLayout(self)
        lay.addWidget(self.image_lbl)
        lay.addWidget(self.ok_button)


class ChatWindow(QMainWindow):
    def __init__(self, gen_model, analysis_model, parent=None):
        super().__init__(parent)

        self.respond_delay = 4000  # delay before get message from Generating model
        self.chat_started = False

        # model to process user messages and generate messages
        self.bot_model = gen_model
        # model to evaluate emotional context
        self.analysis_model = analysis_model

        self.setWindowTitle('ESDS')
        self.setMinimumSize(450, 600)
        central_widget = QWidget()

        self.chat_button = QPushButton("Start Chat")
        self.chat_button.setLayoutDirection(Qt.RightToLeft)
        self.chat_button.pressed.connect(self.chat_button_pressed)

        self.chat_log_model = ChatLogModel()
        chat_log_view = QListView()
        chat_log_view.setModel(self.chat_log_model)
        message_delegate = DrawSpeechBubbleDelegate()
        chat_log_view.setItemDelegate(message_delegate)

        self.user_input_line = QLineEdit()
        self.user_input_line.setPlaceholderText("Press 'Start Chat' to begin chatting...")
        self.user_input_line.textChanged.connect(self.user_typing)
        self.user_input_line.returnPressed.connect(self.massage_entered)

        main_v_box = QVBoxLayout()
        main_v_box.setContentsMargins(0, 2, 0, 10)
        main_v_box.addWidget(self.chat_button, Qt.AlignRight)
        main_v_box.setSpacing(10)
        main_v_box.addWidget(chat_log_view)
        main_v_box.addWidget(self.user_input_line)
        central_widget.setLayout(main_v_box)
        self.setCentralWidget(central_widget)

        # setting up timer for response
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.respond)
        self.message_sent = False

    def chat_button_pressed(self):
        if not self.chat_started:
            self.chat_button.setText("End Chat")
            self.chat_button.setStyleSheet("background: #EC7161")
            self.user_input_line.setPlaceholderText("Type your message and press 'Enter'")
            self.chat_started = True
            # start model thread to process messages in background
            self.bot_model.start()
            self.bot_model.set_default_messages_priority_for(3)  # first 3 messages will be default
            self.chat_log_model.appendMessage(greetings_message, "chatbot")
            self.respond()
        else:
            self.end_chat()

    def respond(self):
        if self.chat_started:
            reply = self.bot_model.get()
            self.chat_log_model.appendMessage(str(reply), "chatbot")
            self.message_sent = False
            self.timer.stop()

    def massage_entered(self):
        user_input = self.user_input_line.text()
        if user_input != "" and self.chat_started:
            self.chat_log_model.appendMessage(user_input, "user")
            self.bot_model.push(user_input)  # push user's messages to model to process
            self.user_input_line.clear()
            self.message_sent = True
            self.timer.start(self.respond_delay)  # reset timer

    def user_typing(self):
        # reset timer to let user write their message
        if self.message_sent:
            self.timer.start(self.respond_delay)

    def end_chat(self):
        choice = QMessageBox.question(self, "End Chat and Get Results",
                                      "Are you sure you want to end the chat now and get results?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if choice == QMessageBox.Yes:
            self.chat_started = False

            # getting analysis from model and display word cloud
            user_messages = ' '.join([m[0] for m in self.chat_log_model.chat_messages if m[1] == 'user'])
            results = self.analysis_model.classify(user_messages)
            WordCloudDialog(results, self).show()

            # clean chat log
            self.chat_log_model.chat_messages = []
            self.user_input_line.setPlaceholderText("Press 'Start Chat' to begin chatting...")
            self.chat_button.setText("Start Chat")
            self.chat_button.setStyleSheet("background: #83E56C")  # Green
        else:
            self.chat_log_model.appendMessage("I thought you were going to leave me.", "chatbot")

    def closeEvent(self, event):
        if self.chat_started:
            choice = QMessageBox.question(self, 'Leave Chat?', "Are you sure you want to leave the chat?",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if choice == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()


if __name__ == "__main__":
    gm = GeneratingModel()
    em = MultipleEmotionsClassifier()
    app = QApplication(sys.argv)
    app.setStyleSheet(style_sheet)
    chat_window = ChatWindow(gen_model=gm, analysis_model=em)
    chat_window.show()
    sys.exit(app.exec_())
