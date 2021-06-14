import os
# import sys
PROJECT_PATH = os.path.abspath('..')


class DatasetsCatalog(object):
    DATASETS = {
        "SAMPLE_HD": {
            "path": os.path.join(PROJECT_PATH, 'datasets/SAMPLE_HD'),
            "h5_dir": os.path.join(PROJECT_PATH, 'datasets/SAMPLE_HD/instructions_h5'),
            "sequence_h5_dir": os.path.join(PROJECT_PATH, 'datasets/sample/sequences_h5')
        }
    }

    @staticmethod
    def get(name, **kwargs):
        if "SAMPLE_HD" in name:
            path = DatasetsCatalog.DATASETS["SAMPLE_HD"]["path"]
            img_dir = os.path.join(path, "images")
            h5_dir = DatasetsCatalog.DATASETS["SAMPLE_HD"]["h5_dir"]
            # sequences_h5_dir = DatasetsCatalog.DATASETS["SAMPLE_HD"]["sequence_h5_dir"]
            vocab_file = 'vocab.json'
            vocab_path = os.path.join(h5_dir, vocab_file)
            instruction_h5 = os.path.join(h5_dir, 'instructions_h5.h5')
            instruction_raw = os.path.join(h5_dir, 'instructions_filtered.json')

            return path, vocab_path, instruction_h5, instruction_raw, img_dir

        raise RuntimeError("Data not available: {}".format(name))
