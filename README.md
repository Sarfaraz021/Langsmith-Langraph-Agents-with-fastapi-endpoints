Dependencies to install:

pip install langchain
pip install pinecone-client
pip install openpyxl
pip install openai
pip install python-dotenv
pip install langchain_community
pip install langchain-openai
pip install docx2txt #to read word documents
pip install unstructured
pip install "unstructured[docx]"
pip install "unstructured[pdf]"

pip install requests
pip install fastapi
pip install uvicorn
pip install --upgrade --quiet langchain-pinecone langchain-openai langchain

->(Optional)
pip install python-jose[cryptography]
pip install motor
pip install motor python-jose passlib
pip install bcrypt

Git Commands for version control:
git add .
git commit -m "define commit"
git push origin 'branch name'
git branch----> to check current branch

Fast app:

To test end points on Swagger UI:

1 - uvicorn app.main:app --reload
---> Type the URL of your running FastAPI application followed by /docs. By default, the URL will be:
http://127.0.0.1:8000/docs swagger UI
