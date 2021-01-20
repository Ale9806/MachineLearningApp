## Machine Learning App with Streamlit
This is an example of a Machine Learning App with Streamlit. Hosted in HEROKU.

### Steps to Host your Application at Heroku
There are many ways to Host your .py file at Heroku. for this example we will implement Github (so the App is updated everytime we push to the Reposetory).

1. First We need to create Our Python files.
2. We need to add the **Procfile**. This file  will run the cnecesary stuff in order to host our app. Remeber that streamlit apps are run in the following way: <br>
```python 
streamlit run [FileName].py
```
Procfile
```
web: sh setup.sh && streamlit run [FileName].py
```
3. Create a **requirements.txt** file that has all the modules you use in you python file
4. Create a **setup.sh** file with the following content:
```bash
mkdir -p ~/.streamlit/ # Create subdirectory (hidden), if it exists already do not raise up an error

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
```
5. Upload your Project to Github
6. Go to [Heroku](https://id.heroku.com/login)
7. Upload your page

You are set to Go!

