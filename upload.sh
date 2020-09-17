rm -r dist
python3 setup.py bdist_wheel
python -m twine upload dist/*
