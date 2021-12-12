from PyQt5.QtWidgets import QApplication, QStyledItemDelegate
from PyQt5.QtCore import Qt, QAbstractListModel, QMargins, QSize, QRect, QPoint
from PyQt5.QtGui import QColor, QImage, QPolygon


class ChatLogModel(QAbstractListModel):
    """Chat log model for QListView"""

    def __init__(self):
        super().__init__()
        self.chat_messages = []

    def rowCount(self, index):
        return len(self.chat_messages)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self.chat_messages[index.row()]

    def appendMessage(self, user_input, user_or_chatbot):
        self.chat_messages.append([user_input, user_or_chatbot])
        self.layoutChanged.emit()


class DrawSpeechBubbleDelegate(QStyledItemDelegate):
    """Draw Delegate for messages in chat log"""

    def __init__(self):
        super().__init__()
        self.image_offset = 5
        self.side_offset, self.top_offset = 40, 5
        self.tail_offset_x, self.tail_offset_y = 30, 0
        self.text_side_offset, self.text_top_offset = 50, 15

    def paint(self, painter, option, index):
        text, user_or_chatbot = index.model().data(index, Qt.DisplayRole)
        image, image_rect = QImage(), QRect()
        color, bubble_margins = QColor(), QMargins()
        tail_points = QPolygon()
        if user_or_chatbot == "chatbot":
            color = QColor("#83E56C")
            bubble_margins = QMargins(self.side_offset, self.top_offset, self.side_offset, self.top_offset)
            tail_points = QPolygon([QPoint(option.rect.x() + self.tail_offset_x, option.rect.center().y()),
                                    QPoint(option.rect.x() + self.side_offset, option.rect.center().y() - 5),
                                    QPoint(option.rect.x() + self.side_offset, option.rect.center().y() + 5)])
        elif user_or_chatbot == "user":
            color = QColor("#38E0F9")
            bubble_margins = QMargins(self.side_offset, self.top_offset, self.side_offset, self.top_offset)
            tail_points = QPolygon([QPoint(option.rect.right() - self.tail_offset_x, option.rect.center().y()),
                                    QPoint(option.rect.right() - self.side_offset, option.rect.center().y() - 5),
                                    QPoint(option.rect.right() - self.side_offset, option.rect.center().y() + 5)])

        painter.setPen(color)
        painter.setBrush(color)
        painter.drawRoundedRect(option.rect.marginsRemoved(bubble_margins), 5, 5)
        painter.drawPolygon(tail_points)
        painter.setPen(QColor("#4A4C4B"))
        text_margins = QMargins(self.text_side_offset, self.text_top_offset, self.text_side_offset,
                                self.text_top_offset)
        painter.drawText(option.rect.marginsRemoved(text_margins), Qt.AlignVCenter | Qt.TextWordWrap, text)

    def sizeHint(self, option, index):
        text, user_or_chatbot = index.model().data(index, Qt.DisplayRole)
        font_size = QApplication.fontMetrics()
        text_margins = QMargins(self.text_side_offset, self.text_top_offset, self.text_side_offset,
                                self.text_top_offset)
        rect = option.rect.marginsRemoved(text_margins)
        rect = font_size.boundingRect(rect, Qt.TextWordWrap, text)
        rect = rect.marginsAdded(text_margins)
        return rect.size()
