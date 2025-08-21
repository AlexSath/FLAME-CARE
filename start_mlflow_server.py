import argparse
import sys, os

from flame.utils import set_up_tracking_server

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--ip")
    args.add_argument("--port")
    args.add_argument("--tracking-direc")
    argv = args.parse_args()

    proc = set_up_tracking_server(
        ip=argv.ip,
        port=argv.port,
        direc=argv.tracking_direc,
        log_path=os.path.abspath("./start_mlflow_server.log")
    )

    print(f"Server started.\nPlease kill the server when finished.")

if __name__ == "__main__":
    main()