
class ExIsTextEngine(TextEngine):
        name = 'ex-is'
        embedding_model = "jonfd/convbert-base-igc-is"
        train_sets = ['train']
        dev_sets = []
        test_sets = []

        labels = ['UNLABELLED']

        input_dir = 'data/example_corpus'

        def iter_files(self, dsets=None, sents_limit=None):
            if dsets is None:
                dsets = self.dsets
            for dset in dsets:
                for filename in listdir(join(self.input_dir, dset)):
                    with open(join(self.input_dir, dset, filename)) as f:
                        for line in f:
                            sentence = line
                            label = 'UNLABELLED'

                            yield sentence, Config(label=label, time=None)


class AlthingiUpptokurTextEngine(TextEngine):
    name = 'alth-upp'
    embedding_model = "jonfd/convbert-base-igc-is"
    train_sets = ['train']
    dev_sets = ['dev']
    test_sets = ['eval']

    labels = ['UNLABELLED']

    input_dir = 'data/example_corpus'

    def iter_files(self, dsets=None, sents_limit=None):
        if dsets is None:
            dsets = self.dsets
        for dset in dsets:
            for filename in listdir(join(self.input_dir, dset)):
                with open(join(self.input_dir, dset, filename)) as f:
                    for line in f:
                        sentence = line
                        label = 'UNLABELLED'

                        yield sentence, Config(label=label, time=None)

