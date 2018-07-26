from distutils.core import setup

setup(
  name = 'mypackage',
  packages = ['nlize'], # this must be the same as the name above
  version = '0.1',
  description = 'Visualization Package for Natural Language Inference',
  author = 'Shusen Liu, Zhimin Li, Tao Li',
  author_email = 'shusenl@sci.utah.edu',
  url = 'https://github.com/shusenl/nlize/', # use the URL to the github repo
  download_url = 'https://github.com/shusenl/nlize//archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['NLP', 'NLI', 'Attention'], # arbitrary keywords
  classifiers = [],
)
