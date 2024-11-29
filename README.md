HOW TO RUN 

1.Download python 

2.Go to terminal and in the proect directory create a vitual enviorment with the following command python -m venv llama_env

3.Activate the enviorment with the following command llama_env\Scripts\activate

4.Install Dependencies :
  1. for systems with nvidia gpus  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  2. for systems without nvidia gpus   pip install torch torchvision torchaudio
  3. pip install transformers
  4. pip install flask

5. Open the main.py file and run it
6. to open the chatbot run the command in terminal  python test_chatbot.py
7. Open browser and navigate to http://127.0.0.1:5000/
