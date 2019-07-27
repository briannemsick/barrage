from barrage import api


class IdentityTransformer(api.RecordTransformer):
    """Default transformer that does nothing (identity transform) that ensures
    every dataset has a transformer.
    """

    def fit(self, records: api.Records):
        pass

    def transform(self, data_record: api.DataRecord) -> api.DataRecord:
        return data_record

    def postprocess(self, score: api.RecordScore) -> api.RecordScore:
        return score

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
