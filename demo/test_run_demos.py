import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )

    logger = logging.getLogger(__name__)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # `run demos`

    This is a test notebook, which runs all the demo notebooks (in headless mode), with the intention to exercise  ~all cells and hence *dpolcat* code.

    We don't currently evaluate output, and consider tests passed if everything just runs without error. The logging provides timing information, which should be monitored. Notebooks finishing very quickly or very slowly indicate a potential problem. And they are subject to the availability of, and connection to, MS Planetary Computer and other external services.
    """)
    return


@app.cell
def _():
    import demo_01_intro

    return (demo_01_intro,)


@app.cell
def _(demo_01_intro):
    logger.info("Starting 'demo_01_intro'.")

    _ = demo_01_intro.app.run()

    logger.info("Finished 'demo_01_intro'.")

    # Expected duration: approx. 10 seconds
    return


@app.cell
def _():
    import demo_02_flood

    return (demo_02_flood,)


@app.cell
def _(demo_02_flood):
    logger.info("Starting 'demo_02_flood'.")

    _ = demo_02_flood.app.run(dict(p_headless=True))

    logger.info("Finished 'demo_02_flood'.")

    # Expected duration: approx. 10 seconds
    return


@app.cell
def _():
    import demo_03_batch_processing

    return (demo_03_batch_processing,)


@app.cell
def _(demo_03_batch_processing):
    logger.info("Starting 'demo_03_batch_processing'.")

    _ = demo_03_batch_processing.app.run(dict(p_headless=True))

    logger.info("Finished 'demo_03_batch_processing'.")

    # Expected duration: approx. 1 minute
    return


@app.cell
def _():
    import demo_04_urban

    return (demo_04_urban,)


@app.cell
def _(demo_04_urban):
    logger.info("Starting 'demo_04_urban'.")

    _ = demo_04_urban.app.run()

    logger.info("Finished 'demo_04_urban'.")

    # Expected duration: approx. 1 minute
    return


if __name__ == "__main__":
    app.run()
