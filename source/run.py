# run.py
from app import create_app
import argparse

app = create_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermAI API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host do servidor")
    parser.add_argument("--port", type=int, default=5000, help="Porta do servidor")
    args = parser.parse_args()
    
    app.run(host=args.host, port=args.port)