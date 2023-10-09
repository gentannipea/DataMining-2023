# DataMining-2023
DataMining (309AA) project for A.Y.2023/2024

## Dataset Download

```bash
curl "http://didawiki.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/dmi/gun-data.zip" -o "dataset/tmp.zip" \
&& unzip "dataset/tmp.zip" -d "dataset/" \
&& rm "dataset/tmp.zip" \
&& mv "dataset/data" "dataset/data-raw"
```

## Requirements installation
It is recomended to use a virtual environment (venv, conda)

### python venv
```bash
python3 -m venv .env \
&& source .env/bin/activate \
&& pip install -r requirements.txt
```