from kennard_stone import train_test_split


def test_train_test_split_with_large_cpu(prepare_large_data):
    _ = train_test_split(*prepare_large_data, test_size=0.2, n_jobs=-1)
