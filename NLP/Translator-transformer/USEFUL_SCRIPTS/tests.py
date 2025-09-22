from transformer import  pos_enc_matrix
from transformer import PositionalEmbedding
from transformer import self_attention
from transformer import cross_attention
from transformer import encoder
from transformer import decoder
from transformer import transformer




import tensorflow as tf

import matplotlib.pyplot as plt
import pickle

def positional_encoding_test():
# Plot the positional encoding matrix
  pos_matrix = pos_enc_matrix(L=2048, d=512)
  assert pos_matrix.shape == (2048, 512)

  plt.pcolormesh(pos_matrix, cmap='RdBu')
  plt.xlabel('Depth')
  plt.ylabel('Position')
  plt.colorbar()
  plt.show()

  
  IMG_DIR = "/content/"
  with open(IMG_DIR + "posenc-2048-512.pickle", "wb") as fp:
      pickle.dump(pos_matrix, fp)


  with open(IMG_DIR +"posenc-2048-512.pickle", "rb") as fp:
    pos_matrix = pickle.load(fp)
  assert pos_matrix.shape == (2048, 512)

  plt.pcolormesh(np.hstack([pos_matrix[:, ::2], pos_matrix[:, 1::2]]), cmap='RdBu')
  plt.xlabel('Depth')
  plt.ylabel('Position')
  plt.colorbar()
  plt.show()

  
  with open(IMG_DIR +"posenc-2048-512.pickle", "rb") as fp:
    pos_matrix = pickle.load(fp)
  assert pos_matrix.shape == (2048, 512)

  plt.plot(pos_matrix[100], alpha=0.66, color="red", label="position 100")
  plt.legend()
  plt.show()


  '''The encoding matrix is useful in the sense that, when you compare two encoding vectors, you can tell how far apart their positions are. 
  The dot-product of two normalized vectors is 1 if they are identical and drops quickly as they move apart. 
  This relationship can be visualized below: 
  '''

  with open(IMG_DIR +"posenc-2048-512.pickle", "rb") as fp:
    pos_matrix = pickle.load(fp)
  assert pos_matrix.shape == (2048, 512)
  # Show the dot product between different normalized positional vectors
  pos_matrix /= np.linalg.norm(pos_matrix, axis=1, keepdims=True)
  p = pos_matrix[789]  # all vectors compare to vector at position 789
  dots = pos_matrix @ p
  plt.plot(dots)
  plt.ylim([0, 1])
  plt.show()


def embedding_layer_test(train_ds):
  vocab_size_it = 20000
  seq_length = 20
  embed_en = 512
  # test the dataset
  for inputs, targets in train_ds.take(1):
      print(inputs["encoder_inputs"])
      embed_en = PositionalEmbedding(sequence_length=seq_length, vocab_size=vocab_size_it, embed_dim= embed_en)
      en_emb = embed_en(inputs["encoder_inputs"])
      print(en_emb.shape)
      print(en_emb._keras_mask) #  matches the input where the position is not zero


def self_attention_layer_test():
  # !conda install -c conda-forge pydotplus -y
    seq_length = 20
    key_dim = 128
    num_heads = 8

    model = self_attention(input_shape=(seq_length, key_dim),
                          num_heads=num_heads, key_dim=key_dim)
    tf.keras.utils.plot_model(model, "self-attention.png",
                              show_shapes=True, show_dtype=True, show_layer_names=True,
                              rankdir='BT', show_layer_activations=True)


    tf.keras.utils.plot_model(model, "ENCODER_self-attention.png",
                              show_shapes=True, show_dtype=True, show_layer_names=True,
                              rankdir='BT', show_layer_activations=True)


def cross_attention_layer_test():
  seq_length = 20
  key_dim = 128
  num_heads = 8

  model = cross_attention(input_shape=(seq_length, key_dim),
                          context_shape=(seq_length, key_dim),
                          num_heads=num_heads, key_dim=key_dim)
  tf.keras.utils.plot_model(model, "cross-attention.png",
                            show_shapes=True, show_dtype=True, show_layer_names=True,
                            rankdir='BT', show_layer_activations=True)

def encoder_layer_test():
  seq_length = 20
  key_dim = 128
  ff_dim = 512
  num_heads = 8

  model = encoder(input_shape=(seq_length, key_dim), key_dim=key_dim, ff_dim=ff_dim,
                  num_heads=num_heads)
  tf.keras.utils.plot_model(model, "encoder.png",
                            show_shapes=True, show_dtype=True, show_layer_names=True,
                            rankdir='BT', show_layer_activations=True)

def decoder_layer_test():
  seq_length = 20
  key_dim = 128
  ff_dim = 512
  num_heads = 8

  model = decoder(input_shape=(seq_length, key_dim), key_dim=key_dim, ff_dim=ff_dim,
                  num_heads=num_heads)
  tf.keras.utils.plot_model(model, "decoder.png",
                            show_shapes=True, show_dtype=True, show_layer_names=True,
                            rankdir='BT', show_layer_activations=True)


def transformer_test():
  seq_len = 20
  num_layers = 4
  num_heads = 8
  key_dim = 128
  ff_dim = 512
  dropout = 0.1
  vocab_size_it = 10000
  vocab_size_eng = 20000
  model = transformer(num_layers, num_heads, seq_len, key_dim, ff_dim,
                      vocab_size_it, vocab_size_eng, dropout)
  tf.keras.utils.plot_model(model, "transformer.png",
                            show_shapes=True, show_dtype=True, show_layer_names=True,
                            rankdir='BT', show_layer_activations=True)
