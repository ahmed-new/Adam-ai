from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from search import retrieve_candidates, ask_ai

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("question")
    session_id = data.get("session_id", "default")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 🔍 Retrieval
    candidates = retrieve_candidates(question, top_n=5)

    # نجمعهم في context
    context = "\n\n".join(candidates)

    # 🤖 Ask Gemini
    answer = ask_ai(context, question, session_id)

    return jsonify({
        "answer": answer,
        "context_used": candidates
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
