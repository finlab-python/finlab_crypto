import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='finlab-crypto',
     version='0.2.15',
     author="FinLab",
     author_email="finlabstaff@gmail.com",
     description="A backtesting framework for crytpo currency",
     long_description=long_description,
   long_description_content_type="text/markdown",
     packages=['finlab_crypto'],
     install_requires=[
        'numpy==1.20.0',
        'numba==0.53.1',
        'pandas==1.1.5',
        'python-binance==0.7.5',
        'pyecharts==1.7.1',
        'vectorbt==0.21.0',
        'statsmodels>=0.10.2',
        'tqdm>=4.41.1',
        'seaborn==0.10.1',
        ],
     python_requires='>=3',
     classifiers=[
         "Programming Language :: Python :: 3",
         "Programming Language :: Python :: 3.4",
         "Programming Language :: Python :: 3.5",
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: OS Independent",
     ],
 )
