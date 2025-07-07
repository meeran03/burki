<div align="center">
  <img src="https://raw.githubusercontent.com/meeran03/burki/main/app/static/logo/light.svg#gh-light-mode-only" alt="Burki Logo" width="300"/>
  <img src="https://raw.githubusercontent.com/meeran03/burki/main/app/static/logo/dark.svg#gh-dark-mode-only" alt="Burki Logo" width="300"/>

  <h1>Burki - The Open-Source Voice AI Platform</h1>
  
  <p><strong>Build and deploy production-ready, multi-tenant AI voice assistants in minutes, not months.</strong></p>

  <p>
    <a href="https://github.com/meeran03/burki/blob/main/LICENSE"><img src="https://img.shields.io/github/license/meeran03/burki?style=for-the-badge" alt="License"></a>
    <a href="https://github.com/meeran03/burki/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=for-the-badge" alt="PRs Welcome"></a>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge" alt="Python 3.11+"></a>
    <a href="https://hub.docker.com/"><img src="https://img.shields.io/badge/Docker-Ready-blueviolet.svg?style=for-the-badge" alt="Docker Ready"></a>
  </p>
</div>

---

**Burki** is a complete, all-in-one platform for creating sophisticated, human-like voice AI agents. It provides the backend infrastructure, a beautiful web UI for management, and all the necessary integrations so you can focus on building great conversational experiences, not on plumbing.

## 📖 Full Documentation

While this README provides a great overview, our full documentation contains detailed guides, API references, and tutorials.

**[➡️ View the Full Documentation](https://docs.burki.dev)**

---

## 🤔 Why Burki?

- **Production-Ready:** Built with a robust, multi-tenant architecture to handle real-world call volume.
- **All-in-One Platform:** Includes a web dashboard, call management, analytics, and API key management out of the box.
- **Extensible & Provider-Agnostic:** Easily integrate with your favorite LLM, TTS, and STT providers.
- **Blazing Fast & Crystal Clear:** Uses real-time audio denoising (RNNoise) and optimized streaming for low-latency conversations.
- **Open-Source & Free:** All the power of a professional voice AI platform, with the flexibility of open-source.

---

## 🚀 Features

- **📞 End-to-End Call Handling:** Full lifecycle management from incoming call to post-call analysis.
- **🤖 Multi-Assistant Support:** Create and manage multiple assistants with unique voices, prompts, and configurations.
- **🌐 Multi-Tenant Architecture:** Support multiple organizations with isolated data and assistants.
- **📊 Real-time Analytics Dashboard:** Monitor call volume, success rates, and performance metrics.
- **🔌 Rich Integrations:**
  - **Telephony:** Twilio
  - **LLM Providers:** OpenAI, Groq, Anthropic, Gemini, and more.
  - **TTS Providers:** ElevenLabs, Deepgram, Inworld, Resemble.
  - **STT Providers:** Deepgram.
- **🎙️ Advanced Audio Processing:**
  - **Real-time Noise Reduction:** Built-in RNNoise for superior audio quality.
  - **Voice Activity Detection (VAD):** Intelligent silence detection for natural turn-taking.
- **💻 Developer Experience:**
  - **RESTful API:** Programmatically manage every aspect of the system.
  - **Webhook Support:** Get real-time notifications for call events.
  - **API Key Management:** Securely manage access for your integrations.

---

## 🛠️ 5-Minute Quick Start (Docker)

Get a full Burki instance running locally with a single command.

**Prerequisites:** Docker & Docker Compose

1.  **Clone the repository:**
   ```bash
   git clone https://github.com/meeran03/burki.git
   cd burki
   ```

2.  **Configure your environment:**
   ```bash
   cp .env.example .env
   ```
    Now, open the `.env` file and add your API keys for Twilio, your chosen LLM, TTS, and STT providers.

3.  **Deploy!**
```bash
chmod +x deploy.sh
./deploy.sh
```
    This script will build the Docker images and start the application and database using Docker Compose.

**🎉 That's it!** Your Burki instance is now running.
- **Web Dashboard:** [http://localhost:8000](http://localhost:8000)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ⚙️ Manual Installation

If you prefer to run the application directly on your host machine:

**Prerequisites:** Python 3.11+, PostgreSQL

1.  **Clone and install dependencies:**
```bash
    git clone https://github.com/meeran03/burki.git
    cd burki
pip install -r requirements.txt
    ```

2.  **Configure your environment:**
    ```bash
    cp .env.example .env
    # Edit the .env file with your credentials and database URL
    ```

3.  **Set up the database:**
    This command runs all necessary database migrations.
```bash
    alembic upgrade head
    ```

4.  **Run the application:**
    ```bash
    # For development
    uvicorn app.main:app --reload

    # For production
    gunicorn app.main:app --bind 0.0.0.0:8000 --worker-class uvicorn.workers.UvicornWorker
    ```
---

## 🤝 Contributing

Contributions are welcome! Please feel free to fork the repository, make changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Maintainer

**Meeran Malik**
- Portfolio: [meeran.dev](https://meeran.dev)
- Twitter/X: [@evolvinginsaan](https://x.com/evolvinginsaan)
- LinkedIn: [Meeran Malik](https://www.linkedin.com/in/meeran-malik-34431316b/)
- GitHub: [@meeran03](https://github.com/meeran03)