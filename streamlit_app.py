import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

runpy.run_path(os.path.join(ROOT, "app", "streamlit_app.py"), run_name="__main__")
