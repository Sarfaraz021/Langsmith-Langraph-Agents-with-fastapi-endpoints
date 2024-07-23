## Dependencies to install:

1. `pip install langchain`
2. `pip install pinecone-client`
3. `pip install openpyxl`
4. `pip install openai`
5. `pip install python-dotenv`
6. `pip install langchain_community`
7. `pip install langchain-openai`
8. `pip install docx2txt` # to read word documents
9. `pip install unstructured`
10. `pip install "unstructured[docx]"`
11. `pip install "unstructured[pdf]"`
12. `pip install requests`
13. `pip install fastapi`
14. `pip install uvicorn`
15. `pip install --upgrade --quiet langchain-pinecone langchain-openai langchain`

### (Optional)

`pip install python-jose[cryptography]`
`pip install motor`
`pip install motor python-jose passlib`
`pip install bcrypt`

## To test api endpoints on Swagger UI:

### Run the following command ensure to you are in 'Varun-AI-Rag-Agent' directory

uvicorn app.main:app --reload

##### Copy and paste this URL on browser, cause your app is runing bydefault on this port 8000:

http://127.0.0.1:8000/docs swagger UI
