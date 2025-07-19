<div align="center">
  <img src="https://burki.dev/static/logo/light.svg#gh-light-mode-only" alt="Burki Logo" width="300"/>
  <img src="https://burki.dev/static/logo/dark.svg#gh-dark-mode-only" alt="Burki Logo" width="300"/>

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

**Burki** is the open-source alternative to vapi.ai that actually delivers on its promises. Unlike proprietary platforms that nickel-and-dime you with complex pricing and poor performance, Burki provides a complete, production-ready voice AI platform with ultra-low latency (0.8-1.2s vs vapi.ai's 4-5s), a beautiful web interface that works, and transparent costs through self-hosting.

## 📖 Try Burki Right Now

<div align="center">

### 🚀 **[Live Demo - burki.dev](https://burki.dev)** 
*Free access for the next 3 months*

### 📚 **[Full Documentation - docs.burki.dev](https://docs.burki.dev)**
*Complete guides, API references, and tutorials*

</div>

**Experience the difference yourself:** See Burki's sub-second latency and beautiful interface in action, then compare it to vapi.ai's sluggish performance. The difference is immediately obvious.

---

## 🤔 Why Burki?

**The Open-Source Alternative to vapi.ai That Actually Works**

Tired of vapi.ai's complex pricing, poor latency, and developer-only complexity? Burki delivers what voice AI platforms should have been from the start.

### **🚀 Superior Performance**
- **Ultra-Low Latency:** 0.8-1.2 seconds vs vapi.ai's 4-5+ seconds
- **Crystal Clear Audio:** Built-in RNNoise for real-time audio denoising
- **Production-Ready:** Multi-tenant architecture that scales to real-world call volumes

### **💰 Transparent & Affordable**
- **No Hidden Costs:** Open-source means no surprise billing or complex pricing tiers
- **All-in-One:** Web dashboard, analytics, and management tools included out of the box
- **Self-Hosted:** Complete control over your costs and data

### **🎯 Actually Usable**
- **Beautiful UI:** A web interface that actually works (unlike vapi.ai's notorious UI issues)
- **Non-Technical Friendly:** Manage assistants without deep developer expertise
- **Complete Platform:** Everything you need in one place, not scattered across multiple services

---

## 📊 Burki vs vapi.ai: The Real Difference

| Feature | Burki (Open-Source) | vapi.ai (Proprietary) |
|---------|---------------------|------------------------|
| **Latency** | 0.8-1.2 seconds | 4-5+ seconds |
| **Pricing** | Free (self-hosted) | $0.07-$0.30/minute + hidden costs |
| **Setup Complexity** | 5-minute Docker deploy | Weeks of API integration |
| **Web Interface** | Beautiful, functional UI | Notorious UI/UX issues |
| **Audio Quality** | Built-in RNNoise denoising | Basic audio processing |
| **All-in-One** | Complete platform | Requires multiple services |
| **Control** | Full data ownership | Vendor lock-in |
| **Transparency** | Open-source code | Black box system |

> **Real User Experience:** *"I was a vapi.ai power user for 2.5 years. The UI never worked properly, latency was terrible (4-5 seconds), and their recent pricing changes would cost my startup $800/month. With Burki, I get sub-second response times and complete control over my costs."*

---

## 🚀 What Burki Can Do (Everything vapi.ai Promises, But Better)

### **📞 Complete Voice AI Platform**
- **End-to-End Call Handling:** Full lifecycle management from incoming call to detailed post-call analysis
- **Multi-Assistant Support:** Create unlimited assistants with unique personalities, voices, and specialized knowledge
- **Real-time Conversations:** WebSocket streaming with 0.8-1.2 second response times (5x faster than vapi.ai)
- **Crystal Clear Audio:** Built-in RNNoise denoising that actually works out of the box

### **🏢 Enterprise-Ready Architecture**
- **Multi-Tenant Design:** Support unlimited organizations with complete data isolation
- **Scalable Infrastructure:** Handle thousands of concurrent calls with auto-scaling
- **Beautiful Web Dashboard:** Manage everything through an interface that actually works
- **Advanced Analytics:** Real-time monitoring, call success rates, and detailed performance metrics

### **🔌 Best-in-Class Integrations**
- **Telephony:** Twilio with WebSocket streaming
- **LLM Providers:** OpenAI, Anthropic, Gemini, xAI, Groq, and custom providers
- **TTS Providers:** ElevenLabs, Deepgram, Inworld, Resemble, OpenAI
- **STT Providers:** Deepgram Nova with confidence scoring
- **Knowledge Base (RAG):** Upload documents to make assistants smarter

### **🎙️ Professional Audio Features**
- **Real-time Noise Reduction:** RNNoise integration for broadcast-quality calls
- **Voice Activity Detection:** Smart silence detection for natural conversations
- **Call Recording:** Automatic recording with transcript storage and search
- **Background Sound Support:** Add ambiance for realistic call environments

### **💻 Developer & Business Friendly**
- **RESTful API:** Complete programmatic control over all platform features
- **Webhook Support:** Real-time notifications for call events and integrations
- **No-Code Assistant Creation:** Build sophisticated voice agents without programming
- **Custom Tool Integration:** Connect to external APIs and databases
- **Secure Authentication:** OAuth, API keys, and role-based access control

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
