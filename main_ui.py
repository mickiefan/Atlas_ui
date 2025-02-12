'''
Author: gengyou.lu 1770591868@qq.com
Date: 2025-01-07 10:28:59
FilePath: /Atlas200_tbps_ui/main_ui.py
LastEditTime: 2025-01-07 10:29:08
Description: Atlas200_tbps_ui
'''
import sys
from PyQt5.QtWidgets import QApplication
from ui_designer.TbpsUiMainWindow import MyMainWindow 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
