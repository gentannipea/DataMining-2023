# DataMining-2023
DataMining (309AA) project for A.Y.2023/2024

## Dataset Download

```bash
mkdir -p "dataset" \
&& curl "http://didawiki.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/dmi/gun-data.zip" -o "dataset/tmp.zip" \
&& unzip "dataset/tmp.zip" -d "dataset/" \
&& rm "dataset/tmp.zip" \
&& mv "dataset/data" "dataset/data-raw"
```

## States population integration dataset

```bash
curl 'https://www.dropbox.com/scl/fi/fu2n5c0qjj2zidwc6ybk7/population.csv?rlkey=i0c1ve16hrbsx8152wl8hyvi4&dl=0' -L > dataset/data-raw/population.csv
```

## Requirements installation
It is recommended to use a virtual environment (venv, conda)

### python venv
```bash
python3 -m venv .env \
&& source .env/bin/activate \
&& pip install -r requirements.txt
```

## Notebooks

- `notebooks/Data_Understanding.ipynb` contains the data understanding phase
- `notebooks/Data_Preparation.ipynb` contains the data preparation phase and exports the cleaned dataset to `dataset/data`
- `notebooks/Clustering.ipynb` contains the clustering phase