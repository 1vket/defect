import re 
import pyopenjtalk

def sentence2phoneSymbol(
  sentence,
  pitch_up='[',
  pitch_down=']',
  accent_border='#',
  sos='^',
  eos='$',
  eos_q='?',
  pose='_',
  drop_unvoiced_vowels=False):
  """
  日本語を音律記号付き音素列に変換します

  Parameters
  ----------
  sentence (str) 
    翻訳したい文
  pitch_up (str) : default '['
    ピッチの上がり位置
  pitch_down (str) : default ']'
    ピッチの下り位置
  accent_border (str) : default '#'
    アクセント句の境界
  sos (str) : default '^'
    文の始まり
  eos (str) : default '$'
    文の終わり
  eos_q (str) : default '?'
    文の終わり（疑問系）
  pose (str) : default '_'
    ポーズ
  drop_unvoiced_vowels (bool) : default False
    無声母音を有声母音に変換
  
  Returns
  -------
  list<chr>
  """

  def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
      return -50
    return int(match.group(1))

  full_contexts = pyopenjtalk.extract_fullcontext(sentence)
  N = len(full_contexts)
  phone_synbol = []

  for i,context in enumerate(full_contexts):
    p3 = re.search(r"\-(.*?)\+", context).group(1)

    if drop_unvoiced_vowels and p3 in "AIUEO":
      p3 = p3.lower()

    if p3 == 'sil':
      if i == 0:
        phone_synbol.append(sos)
      elif i == N-1:
        e3 = numeric_feature_by_regex(r"\!(\d+)_", context)
        if e3 == 0:
          phone_synbol.append(eos)
        elif e3 == 1:
          phone_synbol.append(eos_q)
      continue
    elif p3 == "pau":
      phone_synbol.append(pose)
      continue
    else:
      phone_synbol.append(p3)

    a1 = numeric_feature_by_regex(r"A:([0-9\-]+)\+" , context)
    a2 = numeric_feature_by_regex(r"\+(\d+)\+" , context)
    a3 = numeric_feature_by_regex(r"\+(\d+)/" , context)

    f1 = numeric_feature_by_regex(r"/F:(\d+)_", context)

    next_a2 = numeric_feature_by_regex(r"\+(\d+)\+", full_contexts[i+1]) 

    if a3 == 1 and next_a2 == 1:
      phone_synbol.append(accent_border)
    elif a1 == 0 and next_a2 == a2+1 and a2 != f1:
      phone_synbol.append(pitch_down)
    elif a2 == 1 and next_a2 == 2:
      phone_synbol.append(pitch_up)

  return phone_synbol

def s2ps(sentence, **kwargs):
  return sentence2phoneSymbol(sentence, **kwargs)


if __name__ == "__main__":
  text = "今日の天気は？端，箸，あの橋です"
  print(*sentence2phoneSymbol(text, drop_unvoiced_vowels=True))
  print(s2ps(text, drop_unvoiced_vowels=True))
