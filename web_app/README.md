# Diamond Price Predictor (Streamlit web app)

Serves the model trained in `diamond_price_kfold_full_dataset_10fold.ipynb`.

## Run on your computer
    pip install -r requirements.txt
    streamlit run app.py

## Deploy to the cloud (Streamlit Community Cloud, free)
1. Create a public GitHub repository and upload `app.py`, `requirements.txt`
   and `diamond_price_model.pkl`.
2. Open https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose the repository, branch `main`, main file `app.py`. Under **Advanced settings** pick
   Python 3.13 (the version the model was trained with), then click **Deploy**.
4. Copy the `https://<name>.streamlit.app` link into the first cell of the notebook
   and into the Word report.

If you re-train the model in Colab with different library versions, replace
`diamond_price_model.pkl` with the new file and update the versions in
`requirements.txt` to match the ones printed in Step 1 of the notebook.
