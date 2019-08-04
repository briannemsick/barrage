import logging

# TODO: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass

logging.basicConfig(
    format="Barrage %(asctime)s %(levelname)-5.4s: %(message)s",
    datefmt="%m/%d/%y %I:%M:%S",
    level=logging.INFO,
)
