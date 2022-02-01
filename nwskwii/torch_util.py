import torch

def make_pad_mask(lengths, maxlen=None):
  """
  Make mask for padding frames

  Parameters
  ----------
  lengths (list [Batch_size])
  maxlen (int) : default None

  Returns
  -------
  mask (torch.ByteTensor)
  """
  if not isinstance(lengths, list):
    lengths = lengths.tolist()

  batch_size = int(len(lengths))

  if maxlen is None:
    maxlen = int(max(lengths))

  seq_range = torch.arange(0, maxlen, dtype=torch.int64)
  seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, maxlen)
  seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
  mask = seq_range_expand >= seq_length_expand

  return mask


def make_non_pad_mask(lengths, maxlen=None):
  """
  Make mask for non-padding frames

  Parameters
  ----------
  lengths (list [Batch_size])
  maxlen (int) : default None

  Returns
  -------
  mask (torch.ByteTensor)
  """

  return ~make_pad_mask(lengths, maxlen)


# test code
if __name__ == "__main__":
  lengths = [2,3,4]
  print(make_pad_mask(lengths))
  print(make_pad_mask(lengths, 10))
  print(make_non_pad_mask(lengths))
  print(make_non_pad_mask(lengths, 10))


