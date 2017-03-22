# This file is End to End Memory Networks Applied to MovieQA

def __init__(self, extension='plt', postprocess=False):
    # Filenames & Constants
    ID_MAP_FILE = "id_map_%s.pkl" % extension
    EMBED_FILE = "embed_%s.npz" % extension
    self.SYMBOLS_TO_REMOVE = '"#$%&()*+,/:;<=>@ [\]^_`{|}~-' + "'?!"
    # Get word2vec embeddings
    self.embeddings = []
    with np.load(EMBED_FILE) as f:
        self.embeddings = f['embed']
        print('Embeddings loaded')

        # Get the word to word-id mappings
        self.id_map = self.load_obj(ID_MAP_FILE)

        if postprocess:
            self._postprocess()

        def load_obj(self, name):
        with open(name, 'rb') as f:
        return pickle.load(f)
