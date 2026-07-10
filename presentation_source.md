# Slide 1: Title & Vision
* **Project Name:** Token-Efficient Hybrid AI Gatekeeper
* **Core Function:** An automated, containerized API proxy gateway engineered to act as a financial shield for LLM applications.
* **The Mission:** Dynamically optimizing compute delegation to prevent cloud API budget exhaustion.

# Slide 2: The Core Problem & Architectural Strategy
* **The Budget Burn:** Continuous reliance on premium cloud LLM endpoints quickly depletes developer resources during standard, repetitive tasks.
* **Intelligent Routing Layer:** Intercepts incoming prompts and evaluates logical complexity parameters.
* **Multi-Tier Delegation:**
  * **Local Edge Tier:** Standard tasks route automatically to a zero-cost local Ollama instance running LLaMA.
  * **Premium Cloud Tier:** Complex reasoning tasks scale up securely to remote Fireworks AI endpoints.

# Slide 3: Verified Infrastructure Telemetry
* **Reliability Metric:** Achieved a flawless 100% concurrent request execution success rate during multi-tiered batch evaluations.
* **Network Execution Core:** Fully containerized using Docker with robust cross-platform network bridging (`host.docker.internal`).
* **Performance Speed Integration:**
  * **Average Latency:** 1.4 seconds processing speed under load.
  * **System Throughput:** 32.23 tokens generated per second.

# Slide 4: Team Roles & Strategic Scaling Roadmap
* **Engineering Domain Mappings:**
  * **Sharvin Mhatre:** Core Proxy Infrastructure, Network Bridging, and System Benchmarking.
  * **Archit:** Workload Dataset Engineering and Mock Test Architecture.
  * **Shravani Mayekar:** Frontend Dashboard UI Design and Live Metrics Presentation.
* **Future Scale-Up Roadmap:** Phase 2 production targeting migration from local consumer edge devices to dedicated high-throughput cloud-hosted AMD Instinct™ GPU instances.