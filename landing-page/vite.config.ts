import { defineConfig } from "@lovable.dev/vite-tanstack-config";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CLIENT_ENTRY = path.resolve(__dirname, "src/client.tsx");

// @tanstack/start-plugin-core 1.171.x + Nitro 3.x: the client environment's
// rollup input is set to "virtual:tanstack-start-client-entry" but that virtual
// module is only registered for the SSR environment. Nitro also sets
// rollupOptions.input to index.html for the client env. We override both
// rollupOptions.input and rolldownOptions.input so Rollup uses the real file.
function fixClientRollupInput() {
  return {
    name: "sip:fix-client-rollup-input",
    apply: "build",
    configEnvironment(name, config) {
      if (name !== "client") return;
      if (config.consumer !== "client") return;
      if (!config.build) config.build = {};
      if (!config.build.rollupOptions) config.build.rollupOptions = {};
      if (!config.build.rolldownOptions) config.build.rolldownOptions = {};
      config.build.rollupOptions.input = { index: CLIENT_ENTRY };
      config.build.rolldownOptions.input = { index: CLIENT_ENTRY };
    },
  };
}

export default defineConfig({
  plugins: [fixClientRollupInput()],
  nitro: {
    preset: "vercel",
  },
});