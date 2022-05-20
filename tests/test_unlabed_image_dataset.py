from torch_images import UnlabledImageDataset


def _test(path: str):
    dataset = UnlabledImageDataset(path) 

    assert len(dataset) == 3

    sample = dataset[0]
    assert sample.shape == (3, 256, 256)
    assert sample.min() >= 0.0
    assert sample.max() <= 1.0
    

def test_regular_folder():
    _test("tests/data/folder")


def test_zip_folder():
    _test("tests/data/folder.zip")