"""

This module is the entry point for the MOS Annotator application
Initializes the PyQt5 application, creates the main window, and starts the event loop
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
)

# logging.disable(999999999)

def main():
    """
    Main function to start the MOS Annotator application.
    
    This function creates a QApplication instance, instantiates the MainWindow,
    shows the main window, and executes the application's event loop.
    """
    #creating the Qt application object. This manages application-wide settings.
    app = QApplication(sys.argv)
    logging.info("QApplication created.")

    #creating an instance of the MainWindow, which sets up the GUI.
    window = MainWindow()
    logging.info("MainWindow instance created.")

    #displaying the main window
    window.show()
    logging.info("MainWindow displayed.")

    #execute the application's event loop
    #This call blocks until the application is closed
    exit_code = app.exec_()
    logging.info("Application exited with code %d.", exit_code)

    #Exiting the application with the appropriate exit code
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
