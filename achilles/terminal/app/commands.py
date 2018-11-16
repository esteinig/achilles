import click
from time import sleep

@click.command()
@click.option('--host', '-h', default="localhost")
@click.option('--port', '-p', default="8080")
def app(host, port):
    """Launch server and app for Achilles."""

    click.launch(url=f"https://{host}:{port}/")
    click.o

    try:
        print("Press Ctrl + C to exit.")
        while True:
            sleep(5)
    except (KeyboardInterrupt, SystemError, SystemExit):
        exit(0)



