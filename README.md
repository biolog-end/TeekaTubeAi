# YouTube AI Commenter & Dataset Generator 

This is a personal AI assistant for YouTube. A project that helps you create comments in a unique style, develop a digital persona, and automate the routine.

How it works:
1.  **Select a video** -> **Generate a comment** using AI (Google Gemini).
2.  **Review, edit** -> **Publish**.
3.  Each published comment **updates the dataset**.
4.  The AI is **fine-tuned** on this dataset to match your style better next time!

It results in a kind of eternal engine of content and AI self-learning!

## What can this thing do?

*   **Convenient Flask Web Interface**: Full control right in your browser! No console needed.
*   **Two Modes**:
    *   **Manual**: Full control, processing videos one by one.
    *   **Automatic**: Drop in a bunch of IDs using the provided script, set the batch sizes — and watch the process.
*   **Comment Generation via Google Gemini**: Uses both base models and your own fine-tuned models.
*   **Automatic Dataset Collection**: Every comment you make makes the AI smarter.
*   **"Human-like" Typo Generator**: To prevent comments from looking too perfect, it adds realistic typos — as if you missed a key, skipped a letter, or accidentally swapped them!
*   **Related Video Search**: To comment on several videos on the same topic at once (note: while this function exists, it works less effectively with recent YouTube API updates and consumes many API points).

## How to run?

#### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Configure the API
You need 2 keys from Google. This is the most tedious and difficult part that I can't help with, but it's essential.

*   **YouTube API (OAuth):**
    1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
    2.  Enable the **YouTube Data API v3**.
    3.  Create an **"OAuth client ID"** for a **"Desktop app"**.
    4.  Download the JSON file and rename it to `client_secret.json`.

*   **Gemini API (API Key):**
    1.  Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and get your key.
    2.  Create a `.env` file in the project folder.
    3.  Write one line inside it:
        ```env
        GOOGLE_API_KEY="INSERT_YOUR_KEY_HERE"
        ```

#### 4. Launch!
```bash
python main.py
```
On the first run, a browser window will open to log into your Google account. After that, everything will work automatically.

## How to use?

Run `python main.py` and go to your browser at `http://127.0.0.1:5000`.

*   **Want to do it one by one?** Paste the ID into the first field and click "Start Processing".
*   **Want it automatic?** Paste the IDs into the second field, choose the "batch" size, and click "Start Automatic Processing" (but only after you've built a dataset with your manual responses).

### AI Management Scripts
The package includes several useful scripts:
*   `fine_tune_model.py` - to fine-tune your model on the collected data.
*   `test.py` — to check the current progress.
*   `evaluate_model.py` - to verify how well the model has learned.
*   `delete_model.py` - to delete a model if something went wrong.
