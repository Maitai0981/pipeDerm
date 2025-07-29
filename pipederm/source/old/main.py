import argparse
from api import app
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Porta para execução do servidor")
    parser.add_argument("--debug", action="store_true", help="Modo debug (opcional)")
    args = parser.parse_args()
    
    app.run(
        host="0.0.0.0",
        port=args.port,  # Corrigido: usa args.port
        debug=args.debug,  # Usa o argumento debug
        threaded=True,
        use_reloader=False
    )