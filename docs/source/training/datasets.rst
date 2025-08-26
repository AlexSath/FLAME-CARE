==================
Dataset Management
==================

Dataset management in this repository happens in two steps:

1. Indexing the available images in the provided search folders and assembling dataset metadata associated with a unique id.
2. Configuration of dataset for CARE training (percentile normalization and save to ``.NPZ`` format).

For step one, use ``create_care_dataset.ipynb``.

For step two, use ``care_data_configuration.ipynb``.

After running both steps, an ``.NPZ`` will be present that will be used for training.