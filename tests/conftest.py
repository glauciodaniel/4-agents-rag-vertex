"""
Configuração compartilhada do pytest: path do projeto e fixtures.
"""
import os
import sys
from pathlib import Path

# Raiz do projeto para imports (src.)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    """Evita carregar .env real durante os testes (variáveis são mockadas)."""
    os.environ.setdefault("PYTEST_CURRENT_TEST", "1")
