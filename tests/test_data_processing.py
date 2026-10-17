from src.data_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, load_processed_dataset


def test_processed_dataset_has_expected_columns():
    df = load_processed_dataset()
    assert not df.empty
    assert TARGET_COLUMN in df.columns
    assert set(FEATURE_COLUMNS).issubset(set(df.columns))
    assert df[TARGET_COLUMN].isin([0, 1]).all()
    assert df.isnull().sum().sum() == 0


def test_feature_shape_matches_expectation():
    df = load_processed_dataset()
    assert df.shape[1] == len(FEATURE_COLUMNS) + 1
