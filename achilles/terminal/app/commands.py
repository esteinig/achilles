import click
from time import sleep


@click.command()
@click.option("--host", "-h", default="localhost")
@click.option("--port", "-p", default="8080")
def app(host, port):
    """Launch local server and application for AchillesModel"""

    click.launch(url=f"https://{host}:{port}/")

    try:
        print("Press Ctrl + C to exit.")
        while True:
            sleep(5)
    except KeyboardInterrupt:
        exit(0)
    except (OSError, SystemError, SystemExit):
        exit(1)
