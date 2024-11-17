import pytest

from kennard_stone import train_test_split
from kennard_stone.utils._utils import is_available_gpu


@pytest.mark.skipif(not is_available_gpu(), reason="GPU is not available.")
def test_train_test_split_with_large_gpu(prepare_large_data, get_device):
    _ = train_test_split(*prepare_large_data, test_size=0.2, device=get_device)
