#References:                                                                           #
#https://www.youtube.com/watch?v=jWxNfb7Hng8&t=612s&pp=ygURcHlxdDUgYmVzdCBkZXNpZ24%3D  #
########################################################################################

from PyQt5.QtWidgets import QWidget
from ui.pages.graphic_ui import Ui_Form

class Graphic(QWidget):
    def __init__(self):
        super(Graphic,self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
