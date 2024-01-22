import argparse
import multiprocessing

from whisper_live.server import TranscriptionServer
from whisper_live.client import TranscriptionClient


def start_server(port=9090):
    server = TranscriptionServer()
    server.run(
        "0.0.0.0",
        port,
        #custom_model_path=args.model_path
    )


def start_client(port=9090):
    client = TranscriptionClient(
        "localhost",
        port,
        is_multilingual=True,
        lang="en",
        translate=False,
        model_size="small"
    )
    client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sohkepayin Live")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    # start server and clients in their own processes
    server_process = multiprocessing.Process(target=start_server, args=(args.port,))
    client_process = multiprocessing.Process(target=start_client, args=(args.port,))

    server_process.start()
    client_process.start()

    server_process.join()
    client_process.join()
