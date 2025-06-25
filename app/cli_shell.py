import os
import sys

import click

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.inference import MerchantNameProcessor


@click.command()
def run():
    """A CLI tool that continuously prompts for transaction strings to clean."""
    print("Initializing the Merchant Name Processor...")
    processor = MerchantNameProcessor()
    print("Ready for testing. Type 'exit' or 'quit' to stop.")

    while True:
        text = click.prompt("\nEnter Transaction String")
        if text.lower() in ["exit", "quit"]:
            break

        final_name, confidence = processor.predict(text)

        click.echo("-" * 60)
        click.echo(f"Input Text: {text}")

        if confidence < 0.65:
            click.secho(
                f"Status: Uncertain (Confidence: {confidence:.1%})", fg="yellow"
            )
            click.echo(f"Best Guess: {final_name}")
        else:
            click.secho(
                f"Status: Match Found (Confidence: {confidence:.1%})", fg="green"
            )
            click.echo(f"Official Name: {final_name}")

        click.echo("-" * 60)

    print("Exiting CLI. Goodbye!")


if __name__ == "__main__":
    run()