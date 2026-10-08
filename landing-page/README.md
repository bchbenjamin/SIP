# Proactive Edge-AI Street Safety Landing Page

An interactive showcase of **ASIP_127**, a vandalism-resilient street safety device network designed for edge-AI threat detection, localized deterrence, and ultra-low latency dispatch.

Built with **TanStack Start (React 19)**, **Vite**, and **Tailwind CSS**.

---

## ⚡ Getting Started & Local Setup

Follow these instructions to set up the project locally.

### Prerequisites

Ensure you have [Node.js](https://nodejs.org/) installed (version 18+ is recommended).

### 1. Install Dependencies

Due to peer dependency requirements for the Vite configuration helper and custom Nitro/TanStack setup, install packages using `--legacy-peer-deps`:

```bash
npm install --legacy-peer-deps
```

*Note: If you are using Bun, you can run `bun install` directly.*

### 2. Start the Development Server

To run the application locally on localhost:

```bash
npm run dev
```

By default, the Vite dev server will start the application. Open the URL printed in the terminal (usually `http://localhost:3000` or `http://localhost:5173`) in your browser.

### 3. Build & Preview for Production

To create a production-ready build and run a preview server:

```bash
# Build the application
npm run build

# Preview the built application
npm run preview
```

---

## 📂 Project Structure

Here is a breakdown of the key directories and files:

- **`src/routes/`** - File-based routing structure using TanStack Router:
  - [`__root.tsx`](file:///mnt/WindowsDrive/Fedora/Projects/SIP_landing-page/src/routes/__root.tsx) - Root application shell, containing global layout elements.
  - [`index.tsx`](file:///mnt/WindowsDrive/Fedora/Projects/SIP_landing-page/src/routes/index.tsx) - The Home/Landing page outlining the Edge-AI problem and solution.
  - [`hardware.tsx`](file:///mnt/WindowsDrive/Fedora/Projects/SIP_landing-page/src/routes/hardware.tsx) - The Hardware inspection section details, showing nodes, sensors, and structural architecture.
  - [`response.tsx`](file:///mnt/WindowsDrive/Fedora/Projects/SIP_landing-page/src/routes/response.tsx) - Detailed threat response workflows, active deterrents, and automated escalation.
- **`src/components/`** - Shared React components (such as page transition helpers).
- **`src/styles.css`** - Global styles and Tailwind CSS configurations.
- **`package.json`** - Project dependencies and npm scripts.
- **`vite.config.ts`** - Vite build and plugins configuration.

---

## 🛠️ Tech Stack & Design System

- **Framework**: [TanStack Start](https://tanstack.com/router/latest/docs/start/overview) (React 19)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Animations**: [Framer Motion](https://www.framer.com/motion/) & [tw-animate-css](https://www.npmjs.com/package/tw-animate-css)
