import { createStart } from "@tanstack/react-start";

// Cron wake handling is done at the server entry level (src/server.ts)
// to avoid requestMiddleware chain issues with TanStack Start's router.
export const startInstance = createStart(() => ({}));