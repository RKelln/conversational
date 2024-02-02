import argparse
import logging
logging.basicConfig(level=logging.WARNING)

from whisper_live.server import TranscriptionServer
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sohkepayin Live Server")
    parser.add_argument("--port", type=int, default=9090)

    args = parser.parse_args()

    server = TranscriptionServer()
    server.run(
        "0.0.0.0",
        args.port,
        #custom_model_path=args.model_path
    )

    