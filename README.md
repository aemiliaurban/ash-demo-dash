# Ash

## How to run Ash via Docker
Commands ought to be run from the root of the repo. Alternatively, you can adjust the paths.

### Build the image:
```
docker build . -t ash --no-cache 
```
### Run the container:
```
docker run -p 8050:8050 ash 
```

## How to run Ash locally from the source
Ash supports python3.10. Make sure you have it installed on your machine.
It can be installed with homebrew:
```
brew install python@3.10
```
OPTIONAL: creating a virtual environment (highly recommended):
```
python3.10 -m venv venv
```
Activate the virtual environment
```
source venv/bin/activate
```
Install requirements:
```
pip install -r requirements.txt
```
Run the app:
Before running the app, make sure you are in the ash directory (relative paths).
```
python app.py
```