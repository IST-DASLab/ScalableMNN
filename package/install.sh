pip uninstall -y mnn
rm -rf build
rm -rf dist
rm -rf mnn.egg-info
pip install --no-build-isolation .
