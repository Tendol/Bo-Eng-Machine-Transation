from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# First we create an empty Byte-Pair Encoding model (i.e. not trained model)
tokenizer = Tokenizer(BPE.empty())

# Then we enable lower-casing and unicode-normalization
# The Sequence normalizer allows us to combine multiple Normalizer that will be
# executed in order.
tokenizer.normalizer = Sequence([
    NFKC()
])

# Our tokenizer also needs a pre-tokenizer responsible for converting the input to a ByteLevel representation.
tokenizer.pre_tokenizer = ByteLevel()

# And finally, let's plug a decoder so we can recover from a tokenized input to the original one
tokenizer.decoder = ByteLevelDecoder()


# We initialize our trainer, giving him the details about the vocabulary we want to generate
trainer = BpeTrainer(vocab_size=10, show_progress=True, initial_alphabet=ByteLevel.alphabet())
tokenizer.train(trainer, ["BoTrain.txt"])

print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

# You will see the generated files in the output.
tokenizer.model.save('.')

# Let's tokenizer a simple input
tokenizer.model = BPE.from_files('vocab.json', 'merges.txt')
encoding = tokenizer.encode("བསྟན་")

print("Encoded string: {}".format(encoding.tokens))

decoded = tokenizer.decode(encoding.ids)
print("Decoded string: {}".format(decoded))