import sys
import os
from pathlib import Path
from multiprocessing import Process
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFileDialog, QLineEdit,
    QProgressBar, QScrollArea, QFrame, QTextBrowser, QSizePolicy
)
from qtpy.QtCore import QThread, Signal, QObject, Qt, QSize
import pandas as pd
import numpy as np
import glob
from cellquant.llm_config import load_llm_config, validate_llm_config, format_llm_exception


# Try importing markdown for better rendering, fallback if not installed

try:
    import markdown

    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

# --- STYLESHEET ---
DARK_THEME = """
QMainWindow, QWidget {
    background-color: #121212;
    color: #E0E0E0;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
}

/* Scrollbar Styling */
QScrollBar:vertical {
    border: none;
    background: #1E1E1E;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #424242;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Inputs */
QLineEdit {
    background-color: #252525;
    border: 1px solid #333;
    border-radius: 5px;
    padding: 8px;
    color: #FFF;
    font-size: 14px;
}

QTextEdit {
    background-color: #252525;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 10px;
    color: #FFF;
    font-size: 14px;
}

/* Buttons */
QPushButton {
    border-radius: 6px;
    font-weight: bold;
    font-size: 14px;
    padding: 8px 16px;
}
"""


class ExperimentDataAnalyzer:
    """
    Analyzes experimental data from multiple conditions and prepares
    a comprehensive prompt for GPT API to generate conclusions.
    """

    def __init__(self, base_folder: str):
        self.base_folder = Path(base_folder)
        self.description = ""
        self.experiment_data = {}

    def read_description(self) -> str:
        desc_path = self.base_folder / "description.txt"
        if desc_path.exists():
            with open(desc_path, 'r', encoding='utf-8') as f:
                self.description = f.read().strip()
        else:
            self.description = "No description provided."
        return self.description

    def get_subfolders(self):
        return [f for f in self.base_folder.iterdir() if f.is_dir()]

    def read_csv_files(self, subfolder: Path):
        csv_files = {}
        # Look for direct files
        for csv_file in subfolder.glob("statistic_time*.csv"):
            filename = csv_file.stem
            try:
                time_point = int(filename.replace("statistic_time", ""))
                csv_files[time_point] = pd.read_csv(csv_file)
            except ValueError:
                continue

        # Look for nested files if none found
        if len(csv_files) == 0:
            for csv_file in glob.glob(os.path.join(str(subfolder), '*', "statistic_time*.csv")):
                filename = os.path.basename(csv_file)[:-4]
                try:
                    time_point = int(filename.replace("statistic_time", ""))
                    csv_files[time_point] = pd.read_csv(csv_file)
                except ValueError:
                    continue
        return csv_files

    def calculate_statistics(self, df: pd.DataFrame):
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'median': float(df[col].median()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'n_cells': len(df)

            }
            # Add percentiles
            for p in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
                stats[col][f'{p} percentile'] = float(df[col].quantile(p))
        return stats

    def analyze_condition(self, subfolder: Path):
        condition_name = subfolder.name
        csv_data = self.read_csv_files(subfolder)
        print(f"Analyzing condition '{condition_name}' with {len(csv_data)} time points.")
        if not csv_data:
            return {
                'condition': condition_name,
                'type': 'no_data',
                'message': 'No CSV files found'
            }

        time_points = sorted(csv_data.keys())
        is_timeseries = len(time_points) > 1 or (len(time_points) == 1 and time_points[0] != 0)

        analysis = {
            'condition': condition_name,
            'type': 'time-series' if is_timeseries else 'fixed',
            'time_points': time_points,
            'data': {}
        }

        for time_point, df in csv_data.items():
            analysis['data'][time_point] = {
                'n_cells': len(df),
                'features': list(df.columns),
                'statistics': self.calculate_statistics(df)
            }

        return analysis

    def analyze_all_conditions(self):
        subfolders = self.get_subfolders()
        for subfolder in subfolders:
            condition_analysis = self.analyze_condition(subfolder)
            self.experiment_data[subfolder.name] = condition_analysis

    def format_statistics_summary(self, stats, feature: str) -> str:
        return (f"{feature}: mean={stats['mean']:.3f} ± {stats['std']:.3f}, "
                f"median={stats['median']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}], "
                f"n={stats['n_cells']} cells")

    def generate_gpt_prompt(self, focus_features=None, user_input: str = "") -> str:
        prompt_parts = []
        prompt_parts.append("# Experimental Data Analysis Request\n")

        if user_input:
            prompt_parts.append("# Experiment Description")
            prompt_parts.append(user_input)
            prompt_parts.append("")

        # prompt_parts.append("## Experiment Description")
        # prompt_parts.append(self.description)
        # prompt_parts.append("")

        prompt_parts.append("## Detailed Results by Condition\n")

        for condition_name, analysis in self.experiment_data.items():
            prompt_parts.append(f"### Condition: {condition_name}")
            if analysis['type'] == 'no_data':
                continue

            for time_point in sorted(analysis['data'].keys()):
                tp_data = analysis['data'][time_point]
                prompt_parts.append(f"#### Time Point {time_point}")

                if focus_features:
                    features_to_show = [f for f in focus_features if f in tp_data['statistics']]
                else:
                    features_to_show = list(tp_data['statistics'].keys())

                for feature in features_to_show:
                    if feature in tp_data['statistics']:
                        stats = tp_data['statistics'][feature]
                        prompt_parts.append(f"- {self.format_statistics_summary(stats, feature)}")
                prompt_parts.append("")

        prompt_parts.append("## Analysis Request\n")
        prompt_parts.append(
            "Please provide a comprehensive analysis including key findings, comparative analysis, and biological conclusion and reasonable biological hypothesis behind the experiment. In the final paragraph, use a single paragraph to conclude the conclusion and hypothesis within 150 words.")
        return "\n".join(prompt_parts)


class GPTWorker(QObject):
    """Worker thread for GPT API calls"""
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, api_key, endpoint, api_version, model):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.model = model


    def run_analysis(self, prompt):
        """Run GPT analysis in background thread"""
        try:
            validate_llm_config({
                'api_key': self.api_key,
                'endpoint': self.endpoint,
                'api_version': self.api_version,
                'model': self.model,
            })
            from openai import AzureOpenAI
            self.progress.emit(10)
            client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version
            )
            self.progress.emit(30)
            dialogs = [{"role": "user", "content": prompt}]
            self.progress.emit(50)
            completion = client.chat.completions.create(
                model=self.model,
                messages=dialogs
            )

            self.progress.emit(90)
            ans = completion.choices[0].message.content
            self.progress.emit(100)
            self.finished.emit(ans)
        except Exception as e:
            self.error.emit(format_llm_exception(e, context='Co-Scientists LLM request'))


    def run_analysis_debug(self, prompt):
        """Simulation for testing UI without API costs"""
        try:
            self.progress.emit(10)
            QtCore.QThread.msleep(500)
            self.progress.emit(50)
            QtCore.QThread.msleep(500)

            # Simulate a LONG markdown response
            ans = """# Analysis Result

Here is a **comprehensive** analysis of your data.

## 1. Key Findings
* Observation A: The control group showed stable levels.
* Observation B: The treatment group showed a significant increase.

## 2. Detailed Statistics
| Condition | Mean | Std Dev |
|-----------|------|---------|
| Control   | 10.5 | 2.1     |
| Treated   | 45.2 | 5.3     |

## 3. Biological Interpretation
The increase in structure intensity suggests a strong autophagic response. This correlates with the known pathway activation mechanisms.

""" + ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50)  # Make it long

            self.progress.emit(100)
            self.finished.emit(ans)
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")


class AutoResizingTextBrowser(QTextBrowser):
    """A TextBrowser that adjusts its height to fit content"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setReadOnly(True)
        self.setOpenExternalLinks(True)

        # [FIX 1] Set Size Policy so the layout knows it can grow
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.document().contentsChanged.connect(self.adjust_height)

    def adjust_height(self):
        # [FIX 2] Crucial: Tell the document the actual width of the viewport
        # so it calculates line wrapping correctly.
        width = self.viewport().width()
        if width > 0:
            self.document().setTextWidth(width)

        doc_height = self.document().size().height()
        # Add a little padding
        self.setFixedHeight(int(doc_height + 20))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_height()



class MessageBubble(QFrame):
    """Custom widget for chat message bubbles with Markdown support"""

    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setup_ui(text)

    def showEvent(self, event):
        super().showEvent(event)
        # Find the browser child and force it to adjust
        for child in self.findChildren(AutoResizingTextBrowser):
            child.adjust_height()
    def setup_ui(self, text):
        # Main layout for the row
        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(10, 5, 10, 5)

        # The bubble container
        bubble_frame = QFrame()
        bubble_layout = QVBoxLayout(bubble_frame)
        bubble_layout.setContentsMargins(15, 15, 15, 15)
        bubble_layout.setSpacing(5)

        # 1. Sender Name
        sender_label = QLabel("You" if self.is_user else "Co-Scientists")
        sender_label.setStyleSheet(f"""
            font-weight: bold;
            font-size: 28px;
            color: {'#90CAF9' if self.is_user else '#CE93D8'};
            margin-bottom: 4px;
        """)
        bubble_layout.addWidget(sender_label)

        # 2. Content (Markdown rendered)
        content_browser = AutoResizingTextBrowser()

        # Convert Markdown to HTML if available
        if HAS_MARKDOWN and not self.is_user:
            html_text = markdown.markdown(
                text,
                extensions=['tables', 'fenced_code']
            )
            # Basic CSS for the HTML content
            css = """
            <style>
                body { font-family: 'Segoe UI', sans-serif; font-size: 20px; color: #E0E0E0; }
                h1, h2, h3 { color: #FFF; margin-top: 10px; }
                code { background-color: #424242; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
                pre { background-color: #1E1E1E; padding: 10px; border-radius: 5px; }
                a { color: #64B5F6; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #555; padding: 8px; text-align: left; }
                th { background-color: #333; }
            </style>
            """
            content_browser.setHtml(css + html_text)
        else:
            # User text or fallback
            content_browser.setPlainText(text)
            content_browser.setStyleSheet("font-size: 20px; color: #E0E0E0;")

        # Transparent background for browser so it takes frame color
        content_browser.setStyleSheet(content_browser.styleSheet() + "background: transparent;")

        bubble_layout.addWidget(content_browser)

        # Styling based on role
        if self.is_user:
            bubble_frame.setStyleSheet("""
                QFrame {
                    background-color: #1565C0; /* Darker Blue */
                    border-radius: 15px;
                    border-top-right-radius: 2px;
                    font-size: 25px;
                }
            """)
            # Layout alignment
            row_layout.addStretch()
            row_layout.addWidget(bubble_frame, stretch=0)
            # Limit width of user bubble
            bubble_frame.setMaximumWidth(800)
        else:
            bubble_frame.setStyleSheet("""
                QFrame {
                    background-color: #2D2D2D; /* Dark Grey */
                    border-radius: 15px;
                    border-top-left-radius: 2px;
                    border: 1px solid #424242;
                    font-size: 25px;
                }
            """)
            # Layout alignment
            row_layout.addWidget(bubble_frame, stretch=1)  # Let AI bubble stretch more
            row_layout.addStretch()


class GPTDialogWindow(QMainWindow):
    """Main window for GPT dialog interface"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.analyzer = None
        self.data_folder = None
        self.conversation_history = []

        # Config
        llm_config = load_llm_config()
        self.api_key = llm_config.get('api_key', '')
        self.endpoint = llm_config.get('endpoint', '')
        self.api_version = llm_config.get('api_version', '')
        self.model = llm_config.get('model', 'gpt-4o')

        self.setup_ui()

        self.setWindowTitle("Co-Scientists")
        self.resize(1400, 900)
        self.setStyleSheet(DARK_THEME)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout: Top Bar -> Chat Area -> Input Area
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- 1. HEADER / CONFIG AREA ---
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #1E1E1E; border-bottom: 1px solid #333;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 15, 20, 15)

        title_label = QLabel("Co-Scientists")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #FFF;")

        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("No data folder selected...")
        self.folder_path_edit.setReadOnly(True)
        self.folder_path_edit.setStyleSheet("color: #AAA; border: none; background: transparent;font-size: 20px")

        self.browse_button = QPushButton("📂 Load Data")
        self.browse_button.setStyleSheet("""
            QPushButton { background-color: #333; color: white; border: 1px solid #555; font-size: 25px}
            QPushButton:hover { background-color: #444; }
        """)
        self.browse_button.clicked.connect(self.browse_folder)

        self.export_button = QPushButton("💾 Export")
        self.export_button.setStyleSheet("""
            QPushButton { background-color: #333; color: white; border: 1px solid #555; font-size: 25px}
            QPushButton:hover { background-color: #444; }
        """)
        self.export_button.clicked.connect(self.export_results)

        header_layout.addWidget(title_label)
        header_layout.addSpacing(20)
        header_layout.addWidget(self.folder_path_edit, 1)
        header_layout.addWidget(self.browse_button)
        header_layout.addWidget(self.export_button)

        main_layout.addWidget(header_frame)

        # --- 2. CHAT AREA ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #121212; }")

        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background-color: #121212;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch()  # Pushes messages to bottom
        self.chat_layout.setSpacing(10)
        self.chat_layout.setContentsMargins(20, 20, 20, 20)

        self.scroll_area.setWidget(self.chat_container)
        main_layout.addWidget(self.scroll_area, 1)  # Give this stretch factor 1

        # --- 3. INPUT AREA ---
        input_frame = QFrame()
        input_frame.setStyleSheet("background-color: #1E1E1E; border-top: 1px solid #333;")
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(20, 20, 20, 20)

        self.example_label = QLabel()
        self.example_label.setWordWrap(True)
        # We use HTML <b> tags with inline styles to highlight specific words
        self.example_label.setText(
            "Example: I'm studying <b style='color: #64B5F6'>Lysosome biology</b>. "
            "<b style='color: #64B5F6'>U2OS</b> cell is used. First condition is treated with "
            "<b style='color: #64B5F6'>Rapamycin</b>. Second condition is <b style='color: #64B5F6'>Control</b>. First channel is <b style='color: #64B5F6'>DAPI</b>. Second channel is ""<b style='color: #64B5F6'>Lysotracker</b>. "
            "Focus on "
            "<b style='color: #64B5F6'>Lysosome intensity</b>."
        )
        # Style: Grey text for the normal parts, slightly smaller font
        self.example_label.setStyleSheet("color: #888888; font-size: 24px; margin-bottom: 5px;")

        input_layout.addWidget(self.example_label)

        # Input box
        self.user_input = QTextEdit()
        self.user_input.setPlaceholderText("I'm studying XXX. XXX cell type is treated/KO with XXX. Cell is dyed with XXX (1st channel) and XXX (2nd channel). Focus on XXX feature.")
        self.user_input.setStyleSheet("font-size: 24px;")
        self.user_input.setMaximumHeight(100)
        self.user_input.setMinimumHeight(60)

        # Controls row
        controls_layout = QHBoxLayout()

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #757575; font-size: 24px;")

        self.clear_button = QPushButton("Clear History")
        self.clear_button.setFlat(True)
        self.clear_button.setStyleSheet("color: #EF5350; text-align: left; font-size: 24px")
        self.clear_button.clicked.connect(self.clear_chat)

        self.analyze_button = QPushButton("Send Message")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setCursor(Qt.PointingHandCursor)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                padding: 10px 30px;
                border-radius: 6px;
                font-size: 24px;
            }
            QPushButton:hover { background-color: #2196F3; }
            QPushButton:disabled { background-color: #333; color: #777; }
        """)
        self.analyze_button.clicked.connect(self.run_analysis)

        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.clear_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.analyze_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: none; background: #333; } QProgressBar::chunk { background: #1976D2; }")
        self.progress_bar.setVisible(False)

        input_layout.addWidget(self.user_input)
        input_layout.addLayout(controls_layout)
        input_layout.addWidget(self.progress_bar)

        main_layout.addWidget(input_frame)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", "", QFileDialog.ShowDirsOnly)
        if folder:
            self.data_folder = folder
            self.folder_path_edit.setText(folder)
            self.status_label.setText("Loading data...")

            try:
                self.analyzer = ExperimentDataAnalyzer(folder)
                # self.analyzer.read_description()
                self.analyzer.analyze_all_conditions()

                self.analyze_button.setEnabled(True)
                self.status_label.setText(f"Data Loaded: {len(self.analyzer.experiment_data)} conditions found.")
                self.add_system_message(f"Loaded data from **{os.path.basename(folder)}**. Ready for analysis.")
            except Exception as e:
                self.status_label.setText("Error loading data")
                self.add_system_message(f"Error loading data: {str(e)}")

    def add_system_message(self, text):
        """Adds a small centered system notification"""
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #666; font-style: italic; margin: 10px;")
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, lbl)
        QtCore.QTimer.singleShot(100, self.scroll_to_bottom)

    def add_message(self, text, is_user=True):
        bubble = MessageBubble(text, is_user)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        QApplication.processEvents()
        QtCore.QTimer.singleShot(100, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        sb = self.scroll_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def run_analysis(self):
        if not self.analyzer:
            return

        user_text = self.user_input.toPlainText().strip()
        if not user_text:
            return
        self.add_message(user_text, is_user=True)
        self.user_input.clear()

        # Prepare UI for loading
        self.analyze_button.setEnabled(False)
        self.user_input.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Thinking...")

        # Generate Prompt
        focus_features = [
            'chan2 structure intensity', 'chan1 structure intensity',
            'chan2 structure number', 'chan1 structure number'
        ]
        prompt = self.analyzer.generate_gpt_prompt(user_input=user_text)
        print("Generated Prompt:\n", prompt)
        # Threading
        self.thread = QThread()
        llm_config = load_llm_config()
        self.api_key = llm_config.get('api_key', '')
        self.endpoint = llm_config.get('endpoint', '')
        self.api_version = llm_config.get('api_version', '')
        self.model = llm_config.get('model', 'gpt-4o')
        self.worker = GPTWorker(self.api_key, self.endpoint, self.api_version, self.model)

        self.worker.moveToThread(self.thread)

        # Use debug mode if you don't want to spend tokens, otherwise use run_analysis
        # self.thread.started.connect(lambda: self.worker.run_analysis(prompt))
        self.thread.started.connect(lambda: self.worker.run_analysis(prompt))

        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.start()

    def on_analysis_complete(self, response):
        self.add_message(response, is_user=False)
        self.conversation_history.append({'assistant': response})
        self.reset_ui_state()
        self.status_label.setText("Ready")

    def on_analysis_error(self, error_msg):
        self.add_message(f"**LLM Error**\n\n{error_msg}", is_user=False)
        self.add_system_message(f"Error: {error_msg}")
        self.reset_ui_state()
        self.status_label.setText("LLM error")


    def reset_ui_state(self):
        self.analyze_button.setEnabled(True)
        self.user_input.setEnabled(True)
        self.user_input.setFocus()
        self.progress_bar.setVisible(False)

    def clear_chat(self):
        # Remove widgets from layout (except the stretch at the end)
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.conversation_history.clear()
        self.add_system_message("Conversation cleared.")

    def export_results(self):
        if not self.conversation_history:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export Results", "analysis.md", "Markdown (*.md)")
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                for item in self.conversation_history:
                    f.write(f"\n\n---\n\n{item['assistant']}")
            self.add_system_message(f"Exported to {filename}")


class GPTDialogProcess(Process):
    def __init__(self):
        super().__init__()
        self.daemon = False

    def run(self):
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        window = GPTDialogWindow()
        window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = GPTDialogWindow()
    window.show()
    sys.exit(app.exec_())