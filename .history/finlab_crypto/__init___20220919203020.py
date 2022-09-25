import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from . import crawler
from .strategy import Strategy
from .strategy import Filter

import vectorbt as vbt
import sys
import os

__version__ = '0.2.21'


# set default fees and slippage
vbt.settings.portfolio['init_cash'] = 1000000.0 # in $
vbt.settings.portfolio['fees'] = 0.15       # in %
vbt.settings.portfolio['slippage'] = 0.0  # in %

# has workspace
def check_and_create_dir(dname):
  has_dir = os.path.isdir(dname)
  if not has_dir:
    os.mkdir(dname)

def setup_colab():
  google_drive_connected = os.path.isdir('/content/drive/MyDrive')

  if not google_drive_connected:
    print('|------------------------------')
    print('| Google Drive not connected!  ')
    print('|------------------------------')
    print('|')
    print('| Please connect google drive:')
    from google.colab import drive
    drive.mount('/content/drive')

  # ln -s var
  def ln_dir(path):
    dir = path.split('/')[-1]
    if not os.path.isdir(dir):
        os.symlink(path, dir)

  check_and_create_dir('/content/drive/MyDrive/crypto_workspace')
  # check_and_create_dir('/content/drive/MyDrive/crypto_workspace/strategies')
  check_and_create_dir('/content/drive/MyDrive/crypto_workspace/history')
  # check_and_create_dir('/content/drive/MyDrive/crypto_workspace/filters')
  # ln_dir("/content/drive/MyDrive/crypto_workspace/strategies")
  # ln_dir("/content/drive/MyDrive/crypto_workspace/filters")
  ln_dir("/content/drive/MyDrive/crypto_workspace/history")

def setup():
    IN_COLAB = 'google.colab' in sys.modules
    if IN_COLAB:
        setup_colab()
    else:
        # check_and_create_dir('strategies')
        # check_and_create_dir('filters')
        check_and_create_dir('history')
