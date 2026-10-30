# AI Data Analyst — free deployment version

## Important limitation
This version does not require Ollama or a paid API key. To keep deployment simple and compatible with free hosting, it uses a safe rule-based question interpreter plus pandas for calculations. It supports common analysis questions, but it is not a full large-language-model chatbot and will not understand every possible phrasing.

## Run locally
1. Install Python.
2. Open a terminal in this folder.
3. Create and activate a virtual environment if desired.
4. Install dependencies:
   `pip install -r requirements.txt`
5. Run:
   `streamlit run app.py`

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Open Streamlit Community Cloud and connect your GitHub account.
3. Create an app using the repository, branch, and `app.py`.
4. Deploy and test the generated public URL.

Do not upload private datasets, API keys, or secrets to the repository. Anyone with access to the public app may be able to upload data to it. Do not use confidential personal or business data unless you have appropriate security and access controls.
