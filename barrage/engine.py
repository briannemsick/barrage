import tensorflow as tf

from barrage import api, config, dataset, logger, model, services, solver
from barrage.utils import io_utils, tf_utils


class BarrageModel(object):
    """Class for training the network and scoring records with best performing
    network.

    Args:
        artifact_dir: str, path to artifact directory.
    """

    def __init__(self, artifact_dir):
        self._artifact_dir = artifact_dir

    def train(
        self,
        cfg: dict,
        records_train: api.InputRecords,
        records_validation: api.InputRecords,
    ) -> tf.keras.Model:
        """Train the network.

        Args:
            cfg: dict, config.
            records_train: InputRecords, training records.
            records_validation: InputRecords, validation records.

        Returns:
            tf.keras.Model, trained network.
        """
        logger.info("Starting training")
        tf_utils.reset()
        cfg = config.prepare_config(cfg)

        logger.info(f"Creating artifact directory: {self.artifact_dir}")
        services.make_artifact_dir(self.artifact_dir)
        io_utils.save_json(cfg, "config.json", self.artifact_dir)
        io_utils.save_pickle(cfg, "config.pkl", self.artifact_dir)

        logger.info("Creating datasets")
        ds_train = dataset.RecordDataset(
            artifact_dir=self.artifact_dir,
            cfg_dataset=cfg["dataset"],
            records=records_train,
            mode=api.RecordMode.TRAIN,
            batch_size=cfg["solver"]["batch_size"],
        )
        ds_validation = dataset.RecordDataset(
            artifact_dir=self.artifact_dir,
            cfg_dataset=cfg["dataset"],
            records=records_validation,
            mode=api.RecordMode.VALIDATION,
            batch_size=cfg["solver"]["batch_size"],
        )
        network_params = ds_train.transformer.network_params
        io_utils.save_json(network_params, "network_params.json", self.artifact_dir)
        io_utils.save_pickle(network_params, "network_params.pkl", self.artifact_dir)

        logger.info("Building network")
        net = model.build_network(cfg["model"], network_params)
        model.check_output_names(cfg["model"], net)

        logger.info("Compiling network")
        opt = solver.build_optimizer(cfg["solver"])
        objective = model.build_objective(cfg["model"])
        net.compile(optimizer=opt, **objective)

        logger.info("Creating services")
        callbacks = services.create_all_services(self.artifact_dir, cfg["services"])

        if "learning_rate_reducer" in cfg["solver"]:
            logger.info("Creating learning rate reducer")
            callbacks.append(solver.create_learning_rate_reducer(cfg["solver"]))

        logger.info("Training network")
        net.summary()
        net.fit(
            ds_train,
            validation_data=ds_validation,
            epochs=cfg["solver"]["epochs"],
            steps_per_epoch=cfg["solver"].get("steps"),
            callbacks=callbacks,
            verbose=1,
        )

        return net

    def predict(self, records_score: api.InputRecords) -> api.BatchRecordScores:
        """Score records.

        Args:
            records_score: InputRecords, scoring records.

        Returns:
            BatchRecordScores, scored data records.
        """
        if not hasattr(self, "net"):
            self.load()

        ds_score = dataset.RecordDataset(
            artifact_dir=self.artifact_dir,
            cfg_dataset=self.cfg["dataset"],
            records=records_score,
            mode=api.RecordMode.SCORE,
            batch_size=self.cfg["solver"]["batch_size"],
        )

        network_output = self.net.predict(ds_score, verbose=1)
        scores = [
            ds_score.transformer.postprocess(score)
            for score in dataset.batchify_network_output(
                network_output, self.net.output_names
            )
        ]

        return scores

    def load(self):
        """Load the best performing checkpoint."""

        # Load artifacts needed to recreate the network
        self.cfg = io_utils.load_pickle("config.pkl", self.artifact_dir)
        network_params = io_utils.load_pickle("network_params.pkl", self.artifact_dir)

        # Build network
        self.net = model.build_network(self.cfg["model"], network_params)

        # Load best checkpoint
        path = services.get_best_checkpoint_filepath(self.artifact_dir)
        self.net.load_weights(path).expect_partial()  # not loading optimizer

        return self

    @property
    def artifact_dir(self):
        return self._artifact_dir
