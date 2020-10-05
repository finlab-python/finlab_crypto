import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from . import crawler
from .strategy import Strategy

import vectorbt as vbt
import sys
import os


# set default fees and slippage
vbt.defaults.portfolio['init_capital'] = 100 # in $
vbt.defaults.portfolio['fees'] = 0.001       # in %
vbt.defaults.portfolio['slippage'] = 0.001  # in %

# has workspace
def check_and_create_dir(dname):
  has_dir = os.path.isdir(dname)
  if not has_dir:
    os.mkdir(dname)

def setup_colab():
  google_drive_connected = os.path.isdir('/content/drive/My Drive')

  if not google_drive_connected:
    print('|------------------------------')
    print('| Google Drive not connected!  ')
    print('|------------------------------')
    print('|')
    print('| Please connect google drive and rerun finlab.setup_colab_env')
    print('| *  Files --> Mount Drive --> Connect to google drive')
    return


  # ln -s var
  def ln_dir(path):
    dir = path.split('/')[-1]
    if not os.path.isdir(dir):
        os.symlink(path, dir)

  check_and_create_dir('/content/drive/My Drive/crypto_workspace')
  check_and_create_dir('/content/drive/My Drive/crypto_workspace/strategies')
  check_and_create_dir('/content/drive/My Drive/crypto_workspace/history')
  check_and_create_dir('/content/drive/My Drive/crypto_workspace/filters')
  ln_dir("/content/drive/My Drive/crypto_workspace/strategies")
  ln_dir("/content/drive/My Drive/crypto_workspace/filters")
  ln_dir("/content/drive/My Drive/crypto_workspace/history")

def setup():
    IN_COLAB = 'google.colab' in sys.modules
    if IN_COLAB:
        setup_colab()
    else:
        check_and_create_dir('strategies')
        check_and_create_dir('history')
        check_and_create_dir('filters')

