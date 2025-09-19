import os
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from groq import Groq
from dotenv import load_dotenv

load_dotenv()    
app = Flask(__name__) #Create the Flask application object. __name__ tells Flask where to find static files/templates if needed.
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HTML_FORM = """
<!doctype html>
<title>Mini Tech Challenge</title>
<h2>Customer Call Transcript Analyzer</h2>
<p>Paste any customer transcript below and click "Analyze".</p>
<form method="POST" action="/analyze">
  <textarea name="transcript" rows="6" cols="60" placeholder="Enter transcript here..."></textarea><br><br>
  <input type="submit" value="Analyze">
</form>
"""

# Home Page Route
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_FORM)

# Analyze Transcript Route
@app.route("/analyze", methods=["POST"])
def analyze():
    transcript = request.form.get("transcript")

    if not transcript:
        return "⚠️ Please enter a transcript first!", 400

    # Step 1: Summarization
    summary_prompt = f"Summarize this customer support conversation in 2-3 sentences:\n{transcript}"
    summary_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",  
        messages=[{"role": "user", "content": summary_prompt}]
    )
    summary = summary_response.choices[0].message.content   

    #Sentiment Analysis
    sentiment_prompt = f"Identify if the sentiment of this customer conversation is Positive, Neutral, or Negative:\n{transcript}"
    sentiment_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": sentiment_prompt}]
    )
    sentiment = sentiment_response.choices[0].message.content   

    #Step 3: Save results to CSV 
    recordings_dir = "call_recordings"
    os.makedirs(recordings_dir, exist_ok=True)  # make folder if missing

    csv_file = os.path.join(recordings_dir, "call_analysis.csv")

    # Create a DataFrame for this record
    df = pd.DataFrame(
        [[transcript, summary, sentiment]],
        columns=["Transcript", "Summary", "Sentiment"]
    )

    # Append or create new file
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)

    #Show results to user 
    return jsonify({
        "Transcript": transcript,
        "Summary": summary,
        "Sentiment": sentiment,
        "Saved_to": csv_file
    })


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
