"""
Submodule handling data opening and merging for viewer
"""
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QDialog, QListWidget, QVBoxLayout, QPushButton
from ..run_saves import get_run_cache

TABLES = ['Acquisition', 'Detection', 'Spots', 'Clusters', 'Drift', 'Cell', 'Colocalisation', 'Gene_map']

class PathSelector(QDialog):
    """
    Window for path selection
    """
    def __init__(self, path_map):
        super().__init__()
        self.setWindowTitle("Select Paths")
        self.setGeometry(100, 100, 400, 300)

        self.path_map = path_map  # Dictionary {custom_name: real_path}

        # List Widget
        self.list_widget = QListWidget(self)
        self.list_widget.addItems(path_map.keys())  # Display custom names

        # OK Button
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)
        layout.addWidget(self.ok_button)
        self.setLayout(layout)

    def get_selected_paths(self):
        selected_names = [item.text() for item in self.list_widget.selectedItems()]
        
        res = [self.path_map[name] for name in selected_names]
        if len(res) == 1 :
            return res[0]
        else : return None

def select_path():
    
    run_cached = get_run_cache()
    
    path_list = list(run_cached['RUN_PATH'].unique())
    selection_path = {os.path.basename(path) : path for path in path_list}
    
    app = QApplication([])
    dialog = PathSelector(selection_path)
    if dialog.exec():  # Show dialog and check if OK was pressed
        return dialog.get_selected_paths()
    return None