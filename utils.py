from string import ascii_uppercase

def explore_subject_names(people_names):
  h = dict()
  for c in ascii_uppercase:
    h[c] = sum([p.startswith(c) for p in people_names])
#   print(h)
  return h