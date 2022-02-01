import torch
import torch.nn as nn

class ZoneOutCell(nn.Module):
  def __init__(self, cell, zoneout=0.1):
    """
    Parameters
    ----------
    cell (torch.nn.Module)
    zoneout (float) : default 0.1
    """
    super().__init__()
    self.cell = cell
    self.hidden_size = cell.hidden_size
    self.zoneout = zoneout

  def forward(self, inputs, hidden):
    next_hidden = self.cell(inputs, hidden)
    next_hidden = self._zoneout(hidden, next_hidden, self.zoneout)
    return next_hidden

  def _zoneout(self, h, next_h, prob):
    h_0, c_0 = h
    h_1, c_1 = next_h
    h_1 = self._apply_zoneout(h_0, h_1, prob)
    c_1 = self._apply_zoneout(c_0, c_1, prob)
    return h_1, c_1

  def _apply_zoneout(self, h, next_h, prob):
    if self.training:
      mask = h.new(*h.size()).bernoulli_(prob)
      return mask * h + (1 - mask) * next_h
    else:
      return prob * h + (1 - prob) * next_h

# test code
if __name__ == "__main__":
  lstm = nn.LSTMCell(10, 10)
  lstm = ZoneOutCell(lstm)
  x = torch.rand(3,10)
  y, h = lstm(x, (torch.zeros(3,10), torch.zeros(3,10)))
  

