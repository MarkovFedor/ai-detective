import sys
from pathlib import Path

# чтобы `from src...` работал при запуске pytest и из PyCharm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
