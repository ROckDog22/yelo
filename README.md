# yelo

## Serve a Tensorflow model 
```bash
# Download the Docker image and repo
docker pull wfs2010/yelo:yelo1.0

git clone https://github.com/wfs2010/yelo.git


# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$(pwd)/yelo/static/model/model:/models/model" \
    -e MODEL_NAME=model \
    wfs2010/yelo:yelo1.0 &
```
## Documentation
### Install Set
```bash
pip3 install -r requirements.txt 
```


### Use
```bash
python app.py
cd test
python re.py
```

### output
```angular2
{
    "predictions": [[0.00262068701, 0.000862150046, 0.939066, 0.0220393743, 0.0354117937]]
}

对应5个值'drawings''hentai''neutral''porn''sexy'
```
