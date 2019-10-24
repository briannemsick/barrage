from barrage import api


class IdentityTransformer(api.RecordTransformer):
    """Default transformer that does nothing (identity transform) that ensures
    every dataset has a transformer.
    """

    def fit(self, records: api.Records):
        """Pass - no transform to fit.

        Args:
            records: Records, records.
        """
        pass

    def transform(self, data_record: api.DataRecord) -> api.DataRecord:
        """Identity - return the original data record unchanged.

        Args:
            data_record: DataRecord, data record.

        Returns:
            DataRecord, data record.
        """
        return data_record

    def postprocess(self, score: api.RecordScore) -> api.RecordScore:
        """Identity - return the record score unchanged.

        Args:
            score: RecordScore, record output from net.

        Returns:
            RecordScore, record output from net.
        """
        return score

    def save(self, path: str):
        """Pass - no objects to save.

        Args:
            path: str.
        """
        pass

    def load(self, path: str):
        """Pass - no objects to load.

        Args:
            path: str.
        """
        pass
