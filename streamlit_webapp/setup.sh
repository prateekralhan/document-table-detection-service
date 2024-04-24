mkdir -p ~/.streamlit/
echo "[general]
email = \"Prateek_Ralhan@swissre.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml