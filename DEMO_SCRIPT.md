# POLYMORPH-4 Lite: Sales Demo Script

**Goal:** Demonstrate system reliability, modern UI, and "Crash Proof" architecture to Engineering Managers.
**Duration:** 5-10 Minutes

## Setup (Before the Meeting)

1.  **Start the Backend**:
    ```bash
    # Terminal 1
    cd /path/to/polymorph-lite
    source venv/bin/activate
    python -m retrofitkit.api.server
    ```

2.  **Start the Frontend**:
    ```bash
    # Terminal 2
    cd gui-v2/frontend
    npm run dev
    ```

3.  **Start the AI Service (for the Crash Test)**:
    ```bash
    # Terminal 3
    cd bentoml_service
    ./run_service.sh
    ```
    *(Or ensure the Docker container is running)*

4.  **Open Browser**: Go to `http://localhost:3000`. Log in as `admin@polymorph.com` / `admin123`.

---

## The Demo Flow

### 1. The "First Impression" (1 Minute)
*   **Action**: Show the **System Monitor** page.
*   **Talk Track**:
    > "This is the new POLYMORPH-4 Lite interface. It's built on a modern React stack, fully responsive, and designed for dark-room lab environments."
    > "You can see real-time health status of all connected hardware here. We're currently monitoring the Raman spectrometer, the DAQ unit, and the AI inference engine."

### 2. The "Golden Run" (3 Minutes)
*   **Action**: Navigate to **Workflows**. Select **"Hero Crystallization Demo"**.
*   **Action**: Click **Run**.
*   **Visual**: Switch to the **Dashboard** or **Spectral View**.
*   **Talk Track**:
    > "I'm starting a standard crystallization workflow. This is defined entirely in YAML, so your scientists can version-control their experiments."
    > "Watch the spectral plot. This isn't static data—it's streaming live from the driver at 30Hz. We're using a 'Golden Run' simulation here to show you exactly what a successful polymorph transition looks like."
    > "The system is automatically adjusting the laser power and integration time based on the signal-to-noise ratio."

### 3. The "Crash Test" (The Closer) (2 Minutes)
*   **Action**: While the run is active and data is streaming...
*   **Action**: **Kill the AI Service**.
    *   *If running locally*: `Ctrl+C` in Terminal 3.
    *   *If running Docker*: `docker stop polymorph-ai`.
*   **Visual**: Point to the UI. A warning toast/banner will appear: **"AI Circuit Breaker Open"**.
*   **Crucial Point**: The spectral plot **keeps updating**. The system **does not crash**.
*   **Talk Track**:
    > "I just killed the AI inference engine. In a typical legacy system, the entire application would freeze or crash right now."
    > "But look—the data acquisition continues. The safety interlocks are still active. The 'Circuit Breaker' pattern isolated the failure."
    > "We've designed this to be 'Crash Proof'. Your experiment data is safe, even if the advanced analytics go offline."

### 4. The "Compliance" Check (1 Minute)
*   **Action**: Navigate to **Settings** > **Audit Log**.
*   **Visual**: Show the list of actions (Login, Workflow Start, Circuit Breaker Trip).
*   **Talk Track**:
    > "Every action you just saw was logged here, with a SHA-256 hash chaining it to the previous record. This is fully 21 CFR Part 11 compliant out of the box."

---

## Troubleshooting
- **No Spectra?**: Check if `python -m retrofitkit.api.server` is running.
- **Login Failed?**: Use `admin@polymorph.com` / `admin123`.
- **AI Not Connecting?**: Ensure `bentoml_service` is running on port 3000.
