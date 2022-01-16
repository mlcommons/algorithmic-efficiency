# from https://github.com/pytorch/pytorch/issues/23900#issuecomment-518858050
def cycle(iterable):
  iterator = iter(iterable)
  while True:
    try:
      yield next(iterator)
    except StopIteration:
      iterator = iter(iterable)
