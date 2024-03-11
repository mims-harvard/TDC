echo "Building"
echo "running pip install"
pip install -r $RECIPE_DIR/requirements_conda.txt
echo "installed all pip dependencies.. rest are conda and defined in meta.yaml"
cp -r $RECIPE_DIR/tdc $PREFIX