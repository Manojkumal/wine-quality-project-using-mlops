create env 

```bash
conda create -n wineq python=3.7 -y
```

activate env
```bash
conda activate wineq
```

created a req file

install the req
```bash
pip install -r requirements.txt
```
download the data from 

https://drive.google.com/drive/folders/18zqQiCJVgF7uzXgfbIJ-04zgz1ItNfF5?usp=sharing

```bash
git init
```
```bash
dvc init 
```
```bash
dvc add data_given/winequality.csv
```
```bash
git add .
```
```bash
git commit -m "first commit"
```
```bash
git remote add origin https://github.com/Manojkumal/wine-quality-project-using-mlops.git
git branch -M main
git push origin main
```
tox command -
```bash
tox
```
for rebuilding -
```bash
tox -r
```
pytest command
```bash
pytest -v
```
setup commands -
```bash
pip install -e . 
```
build your own package commands-
````bash
python setup.py sdist bdist_wheel
```